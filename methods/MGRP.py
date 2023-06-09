import time
import numpy as np

from function import Function
from methods.MG import MG


class MGRP(MG):
    def __init__(self, function: Function, tolerance: float, index: int):
        super().__init__(function, tolerance)
        self.index = index

    def second_arg_min(self, _gradient_xk):
        lambda_k: float = self.index + 1 + (1 / (self.k+1))
        norm = (np.linalg.norm(np.array(_gradient_xk)) ** 2)

        return self.arg_min(_gradient_xk) + self.alpha ** 2 * lambda_k * norm

    def algorithm(self) -> None:
        start_time = time.time()

        while self.k < 100:
            if self.found_result():
                self.set_time(start_time)
                break

            _gradient_xk = self.gradient_xk()
            _arg_min = self.second_arg_min(_gradient_xk)
            _alpha_k = self.alpha_k(_arg_min)

            self.xk = self.result(_alpha_k)
            self.all_iterations.append(self.xk)

            self.k += 1

            if self.norm() <= self.tolerance:
                self.set_time(start_time)
                break
