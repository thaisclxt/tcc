import numpy as np

from sympy import symbols, Expr, Derivative


class Function:
    def __init__(self, is_user_input: bool, expression: Expr, global_minimun: np.ndarray):
        self.is_user_input = is_user_input
        self.expression = expression
        self.global_minimun = global_minimun
        self.x, self.y, self.alpha = symbols('x y alpha')
        self.gradient = self.calculate_gradient()

    def calculate_gradient(self) -> list[Expr]:
        grad_y = Derivative(self.expression, self.y, evaluate=True)
        grad_x = Derivative(self.expression, self.x, evaluate=True)

        return [grad_x, grad_y]
