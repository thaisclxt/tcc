import time
import numpy as np

from sympy import simplify
from function import Function
from methods.method import Method


class MGRP(Method):
    def __init__(self, function: Function, tolerance: float, aa: int):
        super().__init__(function, tolerance)
        self.lamda: float = aa + 1 + (1 / (self.k+1))

    def arg_min(self, _gradient_xk):
        f = self.xk - np.array([self.alpha, self.alpha]) * _gradient_xk
        substitution = self.function.expression.subs(
            {self.x: f[0], self.y: f[1]})
        simp = simplify(substitution)
        mgrp = simp + self.alpha ** 2 * self.lamda * \
            (np.linalg.norm(np.array(_gradient_xk)) ** 2)
        return mgrp

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
