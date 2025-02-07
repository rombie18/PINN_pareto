import torch
import torch.nn as nn


class Loss:
    """
    provides the physics loss function class
    """

    # settings read from config (set as class attributes)
    args = ["Re"]

    def __init__(self, pinn, config):

        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])

        # save neural network (weights are updated during training)
        self.pinn = pinn

    def BC(self, X_BC, U_BC):
        """
        Standard MSE Loss for boundary conditions
        """
        U_pred = self.pinn(X_BC)
        loss_BC = nn.functional.mse_loss(U_BC[:, 0:2], U_pred[:, 0:2])
        return loss_BC

    def F(self, X):
        """
        Physics loss based on the Navier-Stokes equation (PINN).
        """
        # Enable gradient tracking
        X.requires_grad_(True)

        # Forward pass through the network
        U = self.pinn(
            X
        )  # Shape: (batch_size, 3), where U[:, 0] = u, U[:, 1] = v, U[:, 2] = p

        u, v, p = (
            U[:, 0],
            U[:, 1],
            U[:, 2],
        )  # Extract velocity (u, v) and pressure (p)

        # Compute first-order derivatives
        grads_u = torch.autograd.grad(
            u, X, grad_outputs=torch.ones_like(u), create_graph=True
        )[
            0
        ]  # Shape: (batch_size, 2)
        grads_v = torch.autograd.grad(
            v, X, grad_outputs=torch.ones_like(v), create_graph=True
        )[0]
        grads_p = torch.autograd.grad(
            p, X, grad_outputs=torch.ones_like(p), create_graph=True
        )[0]

        u_x, u_y = grads_u[:, 0], grads_u[:, 1]  # ∂u/∂x, ∂u/∂y
        v_x, v_y = grads_v[:, 0], grads_v[:, 1]  # ∂v/∂x, ∂v/∂y
        p_x, p_y = grads_p[:, 0], grads_p[:, 1]  # ∂p/∂x, ∂p/∂y

        # Compute second-order derivatives
        u_xx = torch.autograd.grad(
            u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][
            :, 0
        ]  # ∂²u/∂x²
        u_yy = torch.autograd.grad(
            u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][
            :, 1
        ]  # ∂²u/∂y²
        v_xx = torch.autograd.grad(
            v_x, X, grad_outputs=torch.ones_like(v_x), create_graph=True
        )[0][
            :, 0
        ]  # ∂²v/∂x²
        v_yy = torch.autograd.grad(
            v_y, X, grad_outputs=torch.ones_like(v_y), create_graph=True
        )[0][
            :, 1
        ]  # ∂²v/∂y²

        # Navier-Stokes residuals
        res_x = u * u_x + v * u_y + p_x - (u_xx + u_yy) / self.Re
        res_y = u * v_x + v * v_y + p_y - (v_xx + v_yy) / self.Re

        # Compute residual losses
        loss_F_x = torch.mean(res_x**2)
        loss_F_y = torch.mean(res_y**2)

        return loss_F_x, loss_F_y
