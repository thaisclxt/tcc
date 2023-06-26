import numpy as np

from sympy import symbols, Expr, Derivative


class Function:
    """Uma classe que representa função de entrada f(x,y) que será usada no algoritmo."""

    def __init__(self, is_user_input: bool, expression: Expr, global_minimun: np.ndarray):
        self.is_user_input = is_user_input
        """Verifica se a função f(x,y) foi digitada pelo usuário ou se foi escolhida entre as funções de exemplo."""

        self.expression = expression
        """É a função de entrada em formato de uma expressão SymPy."""

        self.global_minimun = global_minimun
        """É o ponto de mínimo global da função f(x,y)."""

        self.x, self.y, self.alpha = symbols('x y alpha')
        """São os símbolos usados ao decorrer do programa de matemática simbólica."""

        self.gradient = self.calculate_gradient()
        """O vetor gradiente da função f(x,y)."""

    def calculate_gradient(self) -> list[Expr]:
        """Calcula o vetor gradiente da função f(x,y).

        Explicação
        ----------
        O vetor gradiente é dado pela derivada de `f(x,y)` em relação à `x` e pela derivada de f(x,y) em relação à `y`.

        Retorno
        -------
        n : `list[Expr]`
            O vetor gradiente da função f(x,y).
        """
        grad_y = Derivative(self.expression, self.y, evaluate=True)
        grad_x = Derivative(self.expression, self.x, evaluate=True)

        return [grad_x, grad_y]
