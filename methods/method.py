import numpy as np
import time

from scipy import optimize
from function import Function
from sympy import Symbol, lambdify


class Method():
    # TODO: documentação
    def __init__(self, function: Function, tolerance: float):
        # TODO: documentação
        self.function = function
        self.tolerance = tolerance
        self.initial_iteration: np.ndarray = np.random.uniform(
            low=0.0, high=10.0, size=2)

        self.k: int = 0
        self.all_iterations: list[np.ndarray] = [self.initial_iteration]
        self.xk: np.ndarray = self.all_iterations[self.k]

        self.x: Symbol = self.function.x
        self.y: Symbol = self.function.y
        self.alpha: Symbol = self.function.alpha

    def gradient_xk(self) -> np.ndarray:
        gradient = self.function.gradient

        a = gradient[0].subs({self.x: self.xk[0], self.y: self.xk[1]})
        b = gradient[1].subs({self.x: self.xk[0], self.y: self.xk[1]})

        return np.array([a, b], dtype=float)

    # TODO: documentação => xk+1
    def result(self, _alpha_k) -> np.ndarray:
        return self.xk - np.array([_alpha_k, _alpha_k]) * self.gradient_xk()

    def alpha_k(self, arg_min):
        f = lambdify(self.alpha, arg_min)

        resultado = optimize.fmin(f, 0, disp=False)
        return resultado[0]

    def get_actual_x(self) -> np.ndarray:
        return

    def set_time(self, start_time: float) -> None:
        self.processing_time = time.time() - start_time

    def found_result(self) -> bool:
        return np.array_equal(self.all_iterations[self.k], self.function.global_minimun)

    def norm(self):
        return np.linalg.norm(self.xk - self.all_iterations[self.k - 1])
