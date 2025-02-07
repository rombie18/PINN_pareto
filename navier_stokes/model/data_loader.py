import numpy as np
import torch
from pyDOE import lhs


class DataLoader:

    # settings read from config (set as class attributes)
    args = ["seed", "L", "Re", "N_BC", "N_col", "N_test"]

    def __init__(self, config, device):

        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])

        # set seed for data sampling
        np.random.seed(self.seed)

        self.device = device

        # aux coefficient for analytical solution
        self.lambd = self.Re / 2 - np.sqrt(
            (self.Re / 2) ** 2 + (2 * np.pi) ** 2
        )

        # domain settings
        self.x_min, self.x_max = -self.L, self.L
        self.y_min, self.y_max = -self.L, self.L
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.X_min = [self.x_min, self.y_min]
        self.X_max = [self.x_max, self.y_max]
        self.X_range = [self.x_range, self.y_range]

    def array2tensor(self, array, exp_dim=True):
        """
        Auxilary function to convert numpy-array to tf-tensor
        expands dimensions if necessary
        """
        if exp_dim:
            array = torch.unsqueeze(array, dim=1)

        # return tf.convert_to_tensor(array, dtype=tf.float32)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def analytical_solution(self, X):
        """
        Returns analytical solution for Kovassznay Flow using PyTorch.
        """
        x = X[:, 0]
        y = X[:, 1]

        exp_term = torch.exp(self.lambd * x / self.L)
        cos_term = torch.cos(2 * torch.pi * y / self.L)
        sin_term = torch.sin(2 * torch.pi * y / self.L)

        u = (1 - exp_term * cos_term) / self.L
        v = (self.lambd / (2 * torch.pi) * exp_term * sin_term) / self.L
        p = -0.5 * torch.exp(2 * self.lambd * x / self.L) / self.L**2

        return self.array2tensor(torch.stack((u, v, p), dim=1), exp_dim=False)

    def regular_grid(self, N=128):
        """
        Provides coordinates and solution on a regular grid
        """
        x = np.linspace(self.x_min, self.x_max, N)
        y = np.linspace(self.y_min, self.y_max, N)

        xx, yy = np.meshgrid(x, y)
        X = self.array2tensor(
            np.vstack((xx.flatten(), yy.flatten())).T, exp_dim=False
        )

        U = self.array2tensor(self.analytical_solution(X), exp_dim=False)

        return X, U

    def sample_BC(self, N=100):
        """
        Provides random samples of BC with N data points at each boundary
        Samples BC from analytical solution
        """
        x_ticks = np.linspace(self.x_min, self.x_max, N)
        y_ticks = np.linspace(self.y_min, self.y_max, N)[1:-1]

        X_left = [[self.x_min, y] for y in y_ticks]
        X_right = [[self.x_max, y] for y in y_ticks]
        X_top = [[x, self.y_max] for x in x_ticks]
        X_bottom = [[x, self.y_min] for x in x_ticks]

        # concatenate and convert to tensor
        X_BC = self.array2tensor(
            X_left + X_right + X_top + X_bottom, exp_dim=False
        )
        U_BC = self.array2tensor(self.analytical_solution(X_BC), exp_dim=False)

        return X_BC, U_BC

    def sample_domain(self, N=1024):
        """
        LHS sampling of coordinates inside function domain
        """

        X = self.array2tensor(
            self.X_min + self.X_range * lhs(2, N), exp_dim=False
        )

        U = self.array2tensor(self.analytical_solution(X), exp_dim=False)

        return X, U

    def sample_datasets(self):
        """
        Provides datasets for BC, collocation and test points
        Uses number of data points as specified in the config file
        """
        X_BC, U_BC = self.sample_BC(N=self.N_BC)
        X_col, U_col = self.sample_domain(N=self.N_col)
        X_test, U_test = self.sample_domain(N=self.N_test)

        datasets = {
            "BC": [X_BC, U_BC],
            "col": [X_col, U_col],
            "test": [X_test, U_test],
        }

        return datasets
