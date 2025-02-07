import torch
import torch.nn as nn


class Loss:
    """
    provides the physics loss function class
    """

    # settings read from config (set as class attributes)
    args = ["kappa"]

    def __init__(self, pinn, config):

        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])

        # store neural network (weights are updated during training)
        self.pinn = pinn

    def u(self, X, u_true):
        """
        Standard MSE Loss for initial and boundary conditions
        """

        u_pred = self.pinn(X)
        loss_u = nn.functional.mse_loss(u_pred, u_true)

        return loss_u

    def F(self, X):
        """
        Physics loss based on diffusion equation
        """
        X.requires_grad_(True)
        u = self.pinn(X)
        u_d = torch.autograd.grad(
            u,
            X,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_t, u_x = u_d[:, 1], u_d[:, 0]
        u_xx = torch.autograd.grad(
            u_x,
            X,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0][:, 0]

        res_F = u_t - self.kappa * u_xx
        loss_F = torch.mean(res_F**2)

        return loss_F
