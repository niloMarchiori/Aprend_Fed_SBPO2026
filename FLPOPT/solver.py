from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from .problem import FederatedLearningProblem


class FLSolver:
    def __init__(self, problem: FederatedLearningProblem,pop_size=150):
        self.problem = problem
        self.algorithm = NSGA2(
            pop_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )

    def solve(self, n_gen=500, seed=1, verbose=False, save_history=False):
        res = minimize(
            self.problem,
            self.algorithm,
            seed=seed, 
            verbose=verbose, 
            save_history=save_history,
            termination=('n_gen', n_gen)
        )
        return res