from sympy import Expr
import numpy as np


class Function:
	def __init__(self, is_user_input: bool, str_function: str, global_minimun: np.ndarray):
		self.is_user_input = is_user_input
		self.str_function = str_function
		self.global_minimun = global_minimun
		self.expression = self.convert_expression()
		self.gradient = self.calculate_gradient()

	def convert_expression(self, expression) -> Expr:
		pass

	def calculate_gradient(self):
		pass
