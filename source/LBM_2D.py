import numpy as np
import matplotlib.pyplot as plt


class LatticeBoltzmann2D:
    def __init__(self, Nx, Ny, U0=0.3, Tau=2.0, nt=1000):
        """
        Initialize the 2D Lattice Boltzmann simulation.
        :param Nx: Number of grid points in the x-direction
        :param Ny: Number of grid points in the y-direction
        :param U0: Initial velocity in the x-direction
        :param Tau: Relaxation time
        :param nt: Number of simulation time steps
        """
        self.Nx, self.Ny = Nx, Ny
        self.U0, self.Tau, self.nt = U0, Tau, nt

        # Number of velocity directions
        self.Nv = 9

        # Lattice directions and weights
        self.XE = np.array([
            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, -1], [1, -1], [-1, 1]
        ])
        self.wk = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

        # Fields
        self.U = np.full((Nx, Ny), U0)
        self.V = np.zeros((Nx, Ny))
        self.Rho = np.full((Nx, Ny), 1.0)
        self.Q = np.zeros((Nx, Ny))
        self.XF = np.zeros((Nx, Ny, self.Nv, 2))
        self.XFEQ = np.zeros((Nx, Ny, self.Nv, 2))

        # Obstacle mask
        self.obstacle_mask = np.zeros((Nx, Ny), dtype=bool)
        self.obstacle_boundary = {
            "U": None,
            "V": None,
            "Q": None
        }

    def add_obstacle(self, mask, U=0, V=0, Q=None):
        """
        Apply an obstacle mask to the simulation.
        :param mask: A boolean mask where True indicates obstacle regions
        """
        self.obstacle_mask = mask
        self.obstacle_boundary = {
            "U" : U,
            "V" : V,
            "Q" : Q
        }


    def equilibrium(self):
        """
        Compute equilibrium distribution functions.
        """
        u2 = self.U ** 2 + self.V ** 2
        for k in range(self.Nv):
            u_ek = self.XE[k, 0] * self.U + self.XE[k, 1] * self.V
            self.XFEQ[:, :, k, 0] = self.wk[k] * self.Rho * (
                1 + 3 * u_ek + 4.5 * u_ek ** 2 - 1.5 * u2
            )
            self.XFEQ[:, :, k, 1] = self.XFEQ[:, :, k, 0] * (self.Q / self.Rho)

    def relaxation(self):
        """
        Relaxation step for the distribution functions.
        """
        self.XF += (self.XFEQ - self.XF) / self.Tau

    def stream(self):
        """
        Streaming step: Shift distribution functions along lattice directions.
        """
        for k in range(self.Nv):
            self.XF[:, :, k, :] = np.roll(self.XF[:, :, k, :], shift=int(self.XE[k, 0]), axis=0)
            self.XF[:, :, k, :] = np.roll(self.XF[:, :, k, :], shift=int(self.XE[k, 1]), axis=1)

    def scalar_moments(self):
        """
        Compute macroscopic moments: Rho, U, V, and Q.
        """
        self.Rho[:, :] = 0
        self.U[:, :] = 0
        self.V[:, :] = 0
        self.Q[:, :] = 0

        for k in range(self.Nv):
            self.Rho += self.XF[:, :, k, 0]
            self.Q += self.XF[:, :, k, 1]

        for k in range(self.Nv):
            self.U += self.XE[k, 0] * self.XF[:, :, k, 0] / self.Rho
            self.V += self.XE[k, 1] * self.XF[:, :, k, 0] / self.Rho

        if not self.obstacle_boundary["U"] is None:
            self.U[self.obstacle_mask] = self.obstacle_boundary["U"]

        if not self.obstacle_boundary["V"] is None:
            self.V[self.obstacle_mask] = self.obstacle_boundary["V"]

        if not self.obstacle_boundary["Q"] is None:
            self.Q[self.obstacle_mask] = self.obstacle_boundary["Q"]


    def boundaries(self):
        """
        Apply boundary conditions.
        """
        self.Q[:, 0] = 1
        self.Q[:, self.Ny - 1] = 0

        self.XF[:, 0, 4, :] = self.XF[:, 0, 3, :]
        self.XF[:, 0, 6, :] = self.XF[:, 0, 5, :]
        self.XF[:, 0, 7, :] = self.XF[:, 0, 8, :]

        self.XF[:, self.Ny - 1, 3, :] = self.XF[:, self.Ny - 1, 4, :]
        self.XF[:, self.Ny - 1, 5, :] = self.XF[:, self.Ny - 1, 6, :]
        self.XF[:, self.Ny - 1, 8, :] = self.XF[:, self.Ny - 1, 7, :]

        self.XF[0, :, :, :] = self.XF[self.Nx - 2, :, :, :]
        self.XF[self.Nx - 1, :, :, :] = self.XF[1, :, :, :]

        self.equilibrium()
        self.XF[:, 0, :, 1] = self.XFEQ[:, 0, :, 1]
        self.XF[:, self.Ny - 1, :, 1] = self.XFEQ[:, self.Ny - 1, :, 1]

    def plot_moments(self):
        """
        Visualize macroscopic moments: U, V, Rho, and Q.
        """
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(self.U.T, aspect='auto')
        plt.colorbar()
        plt.title("U Velocity")

        plt.subplot(2, 2, 2)
        plt.imshow(self.V.T, aspect='auto')
        plt.colorbar()
        plt.title("V Velocity")

        plt.subplot(2, 2, 3)
        plt.imshow(self.Rho.T, aspect='auto', vmin=0.9, vmax=1.1)
        plt.colorbar()
        plt.title("Density Rho")

        plt.subplot(2, 2, 4)
        plt.imshow(self.Q.T, aspect='auto')
        plt.colorbar()
        plt.title("Q")

        plt.tight_layout()
        plt.show()

    def initialize(self):
        """
        Initialize the system.
        """
        # Initialize equilibrium
        self.equilibrium()
        self.XF[:, :, :, :] = self.XFEQ[:, :, :, :]
        self.boundaries()
        self.scalar_moments()

    def plot_profile(self, x_list, variables=["U"]):
        """
        Plot the profiles of given variables at specified X locations.
        :param x_list: List of X indices where the profiles are extracted.
        :param variables: List of variables to plot ('U', 'V', 'Rho', or 'Q').
        """
        # Validate inputs
        for x in x_list:
            if not (0 <= x < self.Nx):
                raise ValueError(f"Each X must be between 0 and {self.Nx - 1}. Got X={x}.")

        for variable in variables:
            if variable not in ["U", "V", "Rho", "Q"]:
                raise ValueError("Variable must be one of 'U', 'V', 'Rho', or 'Q'.")

        # Create subplots for each variable
        num_vars = len(variables)
        fig, axes = plt.subplots(num_vars, 1, figsize=(8, 5 * num_vars), sharex=True)

        if num_vars == 1:
            axes = [axes]  # Ensure axes is a list if there's only one subplot

        for ax, variable in zip(axes, variables):
            data = getattr(self, variable)
            for x in x_list:
                ax.plot(data[x, :], label=f"X={x}")
            ax.set_title(f"{variable} Profile")
            ax.set_xlabel("Y Index")
            ax.set_ylabel(variable)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def simulate(self):
        """
        Main simulation loop.
        """
        # Main time-stepping loop
        for _ in range(self.nt):
            self.equilibrium()
            self.relaxation()
            self.stream()
            self.boundaries()
            self.scalar_moments()

