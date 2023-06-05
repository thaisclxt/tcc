import time
import numpy as np

from sympy import simplify
from function import Function
from methods.method import Method


class MG(Method):
    def __init__(self, function: Function, tolerance: float):
        super().__init__(function, tolerance)

    def arg_min(self, _gradient_xk):
        f = self.xk - np.array([self.alpha, self.alpha]) * _gradient_xk
        return simplify(self.function.expression.subs({self.x: f[0], self.y: f[1]}))

    def algorithm(self):
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
