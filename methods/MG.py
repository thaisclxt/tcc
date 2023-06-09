import time
import numpy as np

from scipy import optimize
from sympy import Symbol, lambdify, simplify
from function import Function


class MG():
    def __init__(self, function: Function, tolerance: float):
        self.function = function
        self.tolerance = tolerance
        self.initial_iteration: np.ndarray = np.random.uniform(low=0.0, high=10.0, size=2)

        self.k: int = 0
        self.all_iterations: list[np.ndarray] = [self.initial_iteration]
        self.xk: np.ndarray = self.all_iterations[self.k]

        self.x: Symbol = self.function.x
        self.y: Symbol = self.function.y
        self.alpha: Symbol = self.function.alpha
        self.processing_time = 0

    def gradient_xk(self) -> np.ndarray:
        gradient = self.function.gradient

        a = gradient[0].subs({self.x: self.xk[0], self.y: self.xk[1]})
        b = gradient[1].subs({self.x: self.xk[0], self.y: self.xk[1]})

        return np.array([a, b], dtype=float)
    
    def arg_min(self, _gradient_xk):
        f = self.xk - np.array([self.alpha, self.alpha]) * _gradient_xk
        return simplify(self.function.expression.subs({self.x: f[0], self.y: f[1]}))

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

    def algorithm(self) -> None:
        start_time = time.time()

        while self.k < 100:
            if self.found_result():
                self.set_time(start_time)
                break

            _gradient_xk = self.gradient_xk()
            _arg_min = self.arg_min(_gradient_xk)
            _alpha_k = self.alpha_k(_arg_min)

            self.xk = self.result(_alpha_k)
            self.all_iterations.append(self.xk)

            self.k += 1

            if self.norm() <= self.tolerance:
                self.set_time(start_time)
                break
