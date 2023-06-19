import time

import numpy as np

from scipy import optimize
from sympy import Symbol, lambdify, simplify
from function import Function


class Method():
    """O método usado no teste.

    Explicação
    ----------
    O método pode ser do tipo Método do Gradiente - MG ou Método do Gradiente com Regularização Proximal - MGRP.

    Atributos
    ---------
    is_MG : bool
        Variável que valida se o método escolhido é o MG".
    function : Function
        Objeto do tipo Function.
    tolerance : float
        A tolerância que será usada como critério de parada para o algoritmo.
    index : int = 0 | 1 | 2
        O índice da iteração do laço de repetição externo.

    initial_iteration : np.ndarray
        O valor da iteração inicial que será sempre atribuído de forma aleatória, a partir do intervalo (0, 10).

    k : int
        O número da iteração que está sendo processada.
    all_iterations : list[np.ndarray]
        Uma lista com os resultados de cada iteração -> todos os `x\u1D4F`.

    x : Symbol
        Um símbolo x da matemática simbólica.
    y : Symbol
        Um símbolo y da matemática simbólica.
    alpha : Symbol
        Um símbolo alpha da matemática simbólica.

    processing_time : float
        O tempo de processamento do algoritmo que será calculado em `self.set_time`.
    """

    def __init__(self, is_MG: bool, function: Function, tolerance: float, index: int):
        self.is_MG = is_MG
        self.function = function
        self.tolerance = tolerance
        self.index = index

        self.initial_iteration: np.ndarray = np.random.uniform(
            low=0.0, high=10.0, size=2)

        self.k: int = 0
        self.all_iterations: list[np.ndarray] = [self.initial_iteration]
        self.xk: np.ndarray = self.all_iterations[self.k]

        self.x: Symbol = self.function.x
        self.y: Symbol = self.function.y
        self.alpha: Symbol = self.function.alpha

        self.processing_time: float = 0.0

    def gradient_xk(self) -> np.ndarray:
        """Calcula o gradiente de `x\u1D4F` na função.

        Explicação
        ----------
        A função gradiente de `Function` já foi calculado na criação do objeto `function`, portanto a função gradiente é invocada e substitui-se o valor de `x\u1D4F`.

        Retorno
        -------
        n : `np.ndarray`
            O resultado do gradiente de `x\u1D4F`.
        """

        gradient = self.function.gradient

        a = gradient[0].subs({self.x: self.xk[0], self.y: self.xk[1]})
        b = gradient[1].subs({self.x: self.xk[0], self.y: self.xk[1]})

        return np.array([a, b], dtype=float)

    def arg_min_MG(self, _gradient_xk):
        f = self.xk - np.array([self.alpha, self.alpha]) * _gradient_xk
        return simplify(self.function.expression.subs({self.x: f[0], self.y: f[1]}))

    def arg_min_MGRP(self, _gradient_xk):
        if self.index == 0:
            lambda_k = 1 / (self.k+1)
        elif self.index == 1:
            lambda_k = 1 + (1 / (self.k+1))
        else:
            lambda_k = 2

        norm = (np.linalg.norm(np.array(_gradient_xk)) ** 2)

        return self.alpha ** 2 * lambda_k * norm

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
            _arg_min = self.arg_min_MG(_gradient_xk)

            if not self.is_MG:
                _arg_min + self.arg_min_MGRP(_gradient_xk)

            _alpha_k = self.alpha_k(_arg_min)

            self.xk = self.result(_alpha_k)
            self.all_iterations.append(self.xk)

            self.k += 1

            if self.norm() <= self.tolerance:
                self.set_time(start_time)
                break
