import torch
import torch.nn as nn
import torch.optim as optim
from model.callback import CustomCallback
from model.data_loader import DataLoader
from model.loss_functions import Loss
from torch.autograd import functional
from torchinfo import summary


class PhysicsInformedNN(nn.Module):
    """
    provides the basic Physics-Informed Neural Network class
    with hard constraints for initial conditions
    """

    # settings read from config (set as class attributes)
    args = [
        "seed",
        "n_hidden",
        "n_neurons",
        "activation",
        "feature_scaling",
        "L",
        "n_epochs",
        "learning_rate",
        "decay_rate",
        "alpha",
    ]

    def __init__(self, config, device, verbose=False):

        # call parent constructor & build NN
        super(PhysicsInformedNN, self).__init__()
        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])

        # set random seed for weights initialization
        torch.manual_seed(self.seed)

        self.device = device

        # builds network architecture
        self.build_network(verbose)
        # create data loader instance
        self.data_loader = DataLoader(config, device)
        # create loss instance
        self.loss = Loss(self, config)
        # create callback instance
        self.callback = CustomCallback(config)

        # system domain for feature scaling
        self.x_min, self.x_max = -self.L, self.L
        self.y_min, self.y_max = -self.L, self.L
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.X_min = torch.tensor(
            [self.x_min, self.y_min], dtype=torch.float32, device=self.device
        )
        self.X_max = torch.tensor(
            [self.x_max, self.y_max], dtype=torch.float32, device=self.device
        )
        self.X_range = torch.tensor(
            [self.x_range, self.y_range],
            dtype=torch.float32,
            device=self.device,
        )
        print("*** PINN build & initialized ***")

    def build_network(self, verbose):
        """
        builds the basic PINN architecture based on
        a PyTorch Sequential model
        """
        layers = []
        # build input layer (x,y)
        layers.append(nn.Linear(2, self.n_neurons))
        layers.append(self.get_activation_function())
        # build hidden layers
        for _ in range(self.n_hidden - 1):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(self.get_activation_function())
        # build 2d linear output layer (Psi, p)
        layers.append(nn.Linear(self.n_neurons, 2))
        self.neural_net = nn.Sequential(*layers)
        self.neural_net.to(self.device)
        # print network summary
        if verbose:
            print(summary(self.neural_net, input_size=(1, 2)))

    def get_activation_function(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        elif self.activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def scale_features(self, X):
        """
        MinMax Feature Scaling to range [-1, 1]
        """
        X_scaled = 2 * (X - self.X_min) / self.X_range - 1
        return X_scaled

    def forward(self, X):
        """
        Override forward method of the (outer) PINN network to use feature scaling
        and hard constrained continuity equation
        """
        if self.feature_scaling:
            X = self.scale_features(X)

        # Enable gradient tracking
        X.requires_grad_(True)

        # Forward pass
        U = self.neural_net(X)

        # Compute gradients of U w.r.t. X
        u_x = torch.autograd.grad(
            U[:, 0], X, grad_outputs=torch.ones_like(U[:, 0]), create_graph=True
        )[0]

        # Extract du/dx and du/dy
        du_dx = u_x[:, 0]  # ∂u/∂x
        du_dy = u_x[:, 1]  # ∂u/∂y

        # Compute v from stream function (negative derivative of Psi w.r.t x)
        v = -du_dx
        u = du_dy

        # Pressure (directly from the second output of the network)
        p = U[:, 1]

        # Stack results
        return torch.stack([u, v, p], dim=1)

    def train(self, mode=True):
        """
        trains the PINN
        """
        super().train(mode)

        # learning rate schedule
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.decay_rate,
        )

        print("Training started...")
        for epoch in range(self.n_epochs):

            # sample and extract training data
            datasets = self.data_loader.sample_datasets()
            X_BC, U_BC = datasets["BC"]
            X_col, _ = datasets["col"]
            X_test, U_test = datasets["test"]

            # perform one train step
            train_logs = self.train_step(X_BC, U_BC, X_col)
            test_logs = self.test_step(X_test, U_test)
            # combine train and test logs
            logs = {**train_logs, **test_logs}
            # provide logs to callback
            self.callback.write_logs(logs, epoch)

            # update learning rate
            if (epoch + 1) % 1000 == 0:
                lr_scheduler.step()

        # save logs and model weights
        self.callback.save_weights(self)
        self.callback.save_logs()
        print("Training finished!")

    def train_step(self, X_BC, U_BC, X_col):
        """
        performs a single SGD training step
        """
        self.optimizer.zero_grad()

        # Data loss (on Boundary)
        loss_U = self.loss.BC(X_BC, U_BC)

        # Physics loss
        loss_F_x, loss_F_y = self.loss.F(X_col)
        loss_F = loss_F_x + loss_F_y

        # weighted mean squared error loss
        loss_train = self.alpha * loss_U + (1 - self.alpha) * loss_F

        # retrieve gradients
        loss_train.backward()
        # perform single GD step
        self.optimizer.step()

        # save logs for recording
        train_logs = {
            "loss_train": loss_train.item(),
            "loss_U": loss_U.item(),
            "loss_F": loss_F.item(),
        }
        return train_logs

    def test_step(self, X_test, U_test):
        """
        Test set performance measures: MSE and rel. L2
        """
        u_test = U_test[:, 0]
        v_test = U_test[:, 1]
        p_test = U_test[:, 2] - U_test[:, 2].mean()
        U_test = torch.stack([u_test, v_test, p_test], dim=1)

        U_pred = self(X_test)
        u_pred = U_pred[:, 0]
        v_pred = U_pred[:, 1]
        p_pred = U_pred[:, 2] - U_pred[:, 2].mean()
        U_pred = torch.stack([u_pred, v_pred, p_pred], dim=1)

        # MSE loss
        loss_test_u = torch.mean((u_test - u_pred) ** 2)
        loss_test_v = torch.mean((v_test - v_pred) ** 2)
        loss_test_p = torch.mean((p_test - p_pred) ** 2)
        loss_test = torch.mean((U_test - U_pred) ** 2)

        # relative L2 norm
        L2_test_u = torch.norm(u_test - u_pred) / torch.norm(u_test)
        L2_test_v = torch.norm(v_test - v_pred) / torch.norm(v_test)
        L2_test_p = torch.norm(p_test - p_pred) / torch.norm(p_test)
        L2_test = torch.norm(U_test - U_pred) / torch.norm(U_test)

        # save logs for recording
        test_logs = {
            "loss_test": loss_test.item(),
            "L2_test": L2_test.item(),
            "L2_test_u": L2_test_u.item(),
            "L2_test_v": L2_test_v.item(),
            "L2_test_p": L2_test_p.item(),
        }
        return test_logs
