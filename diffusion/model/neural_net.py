import torch
import torch.nn as nn
import torch.optim as optim
from model.callback import CustomCallback
from model.data_loader import DataLoader
from model.loss_functions import Loss
from torchinfo import summary


class PhysicsInformedNN(nn.Module):
    """
    provides the basic Physics-Informed Neural Network class
    """

    args = [
        "seed",
        "n_hidden",
        "n_neurons",
        "activation",
        "feature_scaling",
        "kappa",
        "L",
        "lambda_tau",
        "n_epochs",
        "learning_rate",
        "decay_rate",
        "alpha",
    ]

    def __init__(self, config, device, verbose=False):
        super(PhysicsInformedNN, self).__init__()
        for arg in self.args:
            setattr(self, arg, config[arg])
        torch.manual_seed(self.seed)

        self.device = device

        self.build_network(verbose)
        self.data_loader = DataLoader(config, device)
        self.loss = Loss(self, config)
        self.callback = CustomCallback(config)

        self.x_min, self.x_max = 0, self.L
        self.t_min, self.t_max = 0, self.lambda_tau * self.L**2 / self.kappa

        self.x_range = self.x_max - self.x_min
        self.t_range = self.t_max - self.t_min
        self.X_min = torch.tensor(
            [self.x_min, self.t_min], dtype=torch.float32, device=self.device
        )
        self.X_max = torch.tensor(
            [self.x_max, self.t_max], dtype=torch.float32, device=self.device
        )
        self.X_range = torch.tensor(
            [self.x_range, self.t_range],
            dtype=torch.float32,
            device=self.device,
        )

        print("*** PINN build & initialized ***")

    def build_network(self, verbose):
        layers = []
        layers.append(nn.Linear(2, self.n_neurons))
        layers.append(self.get_activation_function())
        for _ in range(self.n_hidden - 1):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(self.get_activation_function())
        layers.append(nn.Linear(self.n_neurons, 1))
        self.neural_net = nn.Sequential(*layers)
        self.neural_net.to(self.device)
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
        X_scaled = (X - self.X_min) / self.X_range
        return X_scaled

    def forward(self, X):
        if self.feature_scaling:
            X = self.scale_features(X)
        return self.neural_net(X)

    def train(self, mode=True):
        super().train(mode)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.decay_rate,
        )

        print("Training started...")
        for epoch in range(self.n_epochs):
            datasets = self.data_loader.sample_datasets()

            X_BC, u_BC = datasets["BC"]
            X_IC, u_IC = datasets["IC"]
            X_col, _ = datasets["col"]
            X_test, u_test = datasets["test"]

            train_logs = self.train_step(X_IC, u_IC, X_BC, u_BC, X_col)
            test_logs = self.test_step(X_test, u_test)
            logs = {**train_logs, **test_logs}
            self.callback.write_logs(logs, epoch)

            if (epoch + 1) % 1000 == 0:
                lr_scheduler.step()

        self.callback.save_weights(self)
        self.callback.save_logs()
        print("Training finished!")

    def train_step(self, X_IC, u_IC, X_BC, u_BC, X_col):
        self.optimizer.zero_grad()

        loss_IC = self.loss.u(X_IC, u_IC)
        loss_BC = self.loss.u(X_BC, u_BC)
        loss_u = loss_IC + loss_BC

        loss_F = self.loss.F(X_col)

        loss_train = self.alpha * loss_u + (1 - self.alpha) * loss_F

        loss_train.backward()
        self.optimizer.step()

        train_logs = {
            "loss_train": loss_train.item(),
            "loss_u": loss_u.item(),
            "loss_F": loss_F.item(),
        }
        return train_logs

    def test_step(self, X_test, u_test):
        with torch.no_grad():
            u_pred = self(X_test)

            loss_test = nn.functional.mse_loss(u_pred, u_test)
            L2_test = torch.norm(u_test - u_pred) / torch.norm(u_test)

            test_logs = {
                "loss_test": loss_test.item(),
                "L2_test": L2_test.item(),
            }
            return test_logs
