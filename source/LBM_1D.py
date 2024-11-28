import numpy as np

class LatticeBoltzmann_1D:
    def __init__(self, N, U=0.1, Tau=2.0, nt=100):
        self.N = N
        self.U = U
        self.Tau = Tau
        self.nt = nt
        self.X_fi_Eq = np.zeros((N, 3))
        self.X_fi = np.zeros((N, 3))
        self.c = np.zeros(N)

        for i in range(0, N):
            # Jump (possible only with LBM)
            if i >= N / 2:
                self.c[i] = 1.
            else:
                self.c[i] = 0.

    def f_equil(self):
        """Compute the equilibrium distribution function."""
        self.X_fi_Eq[:, 0] = self.c[:] * (2. / 3.) * (1. - (3. / 2.) * self.U ** 2)
        self.X_fi_Eq[:, 1] = self.c[:] * (1. / 6.) * (1. + 3 * self.U + (9. / 2.) * self.U ** 2 - (3. / 2.) * self.U ** 2)
        self.X_fi_Eq[:, 2] = self.c[:] * (1. / 6.) * (1. - 3 * self.U + (9. / 2.) * self.U ** 2 - (3. / 2.) * self.U ** 2)

    def f_stream(self):
        """Stream the distribution function."""
        for i in range(self.N - 1, 1, -1):
            self.X_fi[i, 1] = self.X_fi[i - 1, 1]
        for i in range(1, self.N):
            self.X_fi[i - 1, 2] = self.X_fi[i, 2]

    def scalar_moment(self):
        """Compute the scalar moment."""
        self.c[:] = self.X_fi[:, 0] + self.X_fi[:, 1] + self.X_fi[:, 2]

    def initialize(self):
        """Initialize the system."""
        self.f_equil()
        self.X_fi[:, :] = self.X_fi_Eq[:, :]
        self.scalar_moment()

    def simulate(self):
        """Run the Lattice Boltzmann simulation."""
        self.initialize()

        for t in range(self.nt):
            self.f_equil()
            self.X_fi[:, :] = self.X_fi[:, :] + (self.X_fi_Eq[:, :] - self.X_fi[:, :]) / self.Tau
            self.f_stream()
            self.X_fi[0, :] = 0
            self.X_fi[self.N - 2, :] = self.X_fi[self.N - 1, :]
            self.scalar_moment()
