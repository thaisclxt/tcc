import time

import numpy as np

from scipy import optimize
from sympy import Symbol, Expr, lambdify, simplify
from function import Function


class Method():
    """O método usado no teste.

    Explicação
    ----------
    O método pode ser do tipo Método do Gradiente - MG ou Método do Gradiente com Regularização Proximal - MGRP.        
    """

    def __init__(self, is_MG: bool, function: Function, tolerance: float, index: int):
        self.is_MG = is_MG
        """Variável que valida se o método escolhido é o MG."""

        self.function = function
        """Instância da classe `Function`."""

        self.tolerance = tolerance
        """A tolerância que será usada como critério de parada para o algoritmo."""

        self.index = index
        """O índice da iteração do laço de repetição externo, podendo ser 0 | 1 | 2."""

        self.initial_iteration: np.ndarray = np.random.uniform(
            low=0.0, high=10.0, size=2)
        """O valor da iteração inicial que será sempre atribuído de forma aleatória, a partir do intervalo (0, 10)."""

        self.k: int = 0
        """O número da iteração que está sendo processada."""

        self.all_iterations: list[np.ndarray] = [self.initial_iteration]
        """Uma lista com os resultados de cada iteração -> todos os `x\u1D4F`."""

        self.xk: np.ndarray = self.all_iterations[self.k]
        """O valor do `x\u1D4F` atual."""

        self.x: Symbol = self.function.x
        """Um símbolo x da matemática simbólica."""

        self.y: Symbol = self.function.y
        """Um símbolo y da matemática simbólica."""

        self.alpha: Symbol = self.function.alpha
        """Um símbolo alpha da matemática simbólica."""

        self.processing_time: float = 0.0
        """O tempo de processamento do algoritmo que será calculado em `self.set_time`."""

    def gradient_xk(self) -> np.ndarray:
        """Calcula o gradiente de `f(x\u1D4F)`.

        Explicação
        ----------
        O vetor gradiente de `f(x,y)` já foi calculado na criação do objeto `function`, portanto basta substituir os valor de `x\u1D4F`.

        Retorno
        -------
        n : `np.ndarray`
            O resultado do gradiente de `x\u1D4F`.
        """

        gradient = self.function.gradient

        a = gradient[0].subs({self.x: self.xk[0], self.y: self.xk[1]})
        b = gradient[1].subs({self.x: self.xk[0], self.y: self.xk[1]})

        return np.array([a, b], dtype=float)

    def multiply_tow_squared_values(self, _gradient_xk):
        if self.index == 0:
            lambda_k = 1 / (self.k+1)
        elif self.index == 1:
            lambda_k = 1 + (1 / (self.k+1))
        else:
            lambda_k = 2

        norm = (np.linalg.norm(np.array(_gradient_xk)) ** 2)

        return self.alpha ** 2 * lambda_k * norm

    def subtract_tow_vectors(self, _gradient_xk) -> np.ndarray:
        return self.xk - np.array([self.alpha, self.alpha]) * _gradient_xk

    def substitute_x_y(self, _gradient_xk) -> Expr:
        f = self.subtract_tow_vectors(_gradient_xk)
        return self.function.expression.subs({self.x: f[0], self.y: f[1]})

    def arg_min(self, _gradient_xk):
        function = self.substitute_x_y(_gradient_xk)

        if not self.is_MG:
            function += self.multiply_tow_squared_values(_gradient_xk)

        simplified_function: Expr = simplify(function)

        f = lambdify(self.alpha, simplified_function)
        minimize = optimize.fmin(f, 0, disp=False)

        return minimize[0]

    # TODO: documentação => xk+1
    def result(self, _gradient_xk) -> np.ndarray:
        alpha_k = self.arg_min(_gradient_xk)

        return self.xk - np.array([alpha_k, alpha_k]) * self.gradient_xk()

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

            self.xk = self.result(_gradient_xk)
            self.all_iterations.append(self.xk)

            self.k += 1

            if self.norm() <= self.tolerance:
                self.set_time(start_time)
                break
