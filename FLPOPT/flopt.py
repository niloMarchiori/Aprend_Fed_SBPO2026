from .solver import FLSolver
from .problem import FederatedLearningProblem


class FLPOPT:
    def __init__(self, N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev, T_min=1.0, T_max=500.0):
        self.problem = FederatedLearningProblem(N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev, T_min, T_max)
        self.solver = FLSolver(self.problem)

    def solve(self, n_gen=500, seed=1, verbose=False,save_history=False):
        return self.solver.solve(n_gen=n_gen, seed=seed, verbose=verbose, save_history=save_history)