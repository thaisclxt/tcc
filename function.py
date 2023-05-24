from sympy import Expr
import numpy as np


class Function:
    def __init__(self, is_user_input: bool, expression: Expr, global_minimun: np.ndarray):
        self.is_user_input = is_user_input
        self.expression = expression
        self.global_minimun = global_minimun
        self.gradient = self.calculate_gradient()

    def calculate_gradient(self):
        pass
