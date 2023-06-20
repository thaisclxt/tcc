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

    def MGRP_equation(self, _gradient_xk) -> np.ndarray:
        """Calcula \u03b1² λk \u2225 ∇f(x\u1D4F) \u2225².

        Explicação
        ----------
        É a parte da equação do MGRP que se difere da equação do MG.

        Retorno
        -------
        n : `np.ndarray`
            O resultado da multiplicação dos 3 valores.
        """

        if self.index == 0:
            lambda_k = 1 / (self.k+1)
        elif self.index == 1:
            lambda_k = 1 + (1 / (self.k+1))
        else:
            lambda_k = 2

        norm = (np.linalg.norm(np.array(_gradient_xk)) ** 2)

        return self.alpha ** 2 * lambda_k * norm

    def subtract_tow_vectors(self, _gradient_xk) -> np.ndarray:
        """Subtrai dois vetores.

        Explicação
        ----------
        É a subtração de `x\u1D4F` com `\u03b1 ∇f(x\u1D4F)`.

        Retorno
        -------
        n : `np.ndarray`
            O resultado da subtração.
        """

        return self.xk - np.array([self.alpha, self.alpha]) * _gradient_xk

    def substitute_x_y(self, _gradient_xk) -> Expr:
        """Substitui os valores de `x` e de `y` na na função principal.

        Explicação
        ----------
        É a substituição dos valores encontrados em `subtract_tow_vector()` na função principal f(x, y).

        Retorno
        -------
        n : `Expr`
            Retorna uma expressão.
        """

        f = self.subtract_tow_vectors(_gradient_xk)
        return self.function.expression.subs({self.x: f[0], self.y: f[1]})

    def arg_min(self, _gradient_xk) -> float:
        """Calcula o argumento mínimo da expressão resultante de `substitute_x_y()`.

        Explicação
        ----------
        Depois de substituir os valores de `x` e `y` através de `substitute_x_y()`, o algoritmo verifica qual o tipo do método a ser processado.

        Caso seja MGRP, a equação `MGRP_equation()` é somada à função.

        Em seguida, a função (expressão SymPy) é simplificada e convertida em uma função que permite uma avaliação numérica rápida.

        Então, o arugmento mínimo é calculado através do método `optimize.fmin()`.

        Retorno
        -------
        n : `float`
            Retorna o valor do argumento mínimo da expressão.
        """

        function = self.substitute_x_y(_gradient_xk)

        if not self.is_MG:
            function += self.MGRP_equation(_gradient_xk)

        simplified_function: Expr = simplify(function)

        f = lambdify(self.alpha, simplified_function)
        minimize = optimize.fmin(f, 0, disp=False)

        return minimize[0]

    def result(self, _gradient_xk) -> np.ndarray:
        """Calcula o valor de `x\u1D4F⁺¹`.

        Explicação
        ----------
        Depois de calcular o valor \u03b1\u1D4F em `arg_min()`, é calculado o valor de `x\u1D4F⁺¹`.

        Retorno
        -------
        n : `np.ndarray`
            Retorna o valor resultante da iteração k.
        """

        alpha_k = self.arg_min(_gradient_xk)

        return self.xk - np.array([alpha_k, alpha_k]) * self.gradient_xk()

    def set_time(self, start_time: float) -> None:
        """Calcula o tempo de processamento do algoritmo.

        Parâmetro
        ---------
        start_time : float
            Tempo de início.
        """
        self.processing_time = time.time() - start_time

    def found_result(self) -> bool:
        """Verifica se encontrou o resultado esperado.

        Explicação
        ----------
        Compara o resultado de `x\u1D4F` com o valor do mínimo global `x*`.

        Retorno
        -------
        n : bool
            Retorna true se forem iguais e false se não.
        """

        return np.array_equal(self.all_iterations[self.k], self.function.global_minimun)

    def norm(self) -> np.ndarray:
        """Calcula a norma entre dois valores

        Explicação
        ----------
        Calcula a distância entre o resultado da iteração atual (x\u1D4F) e o resultado da iteração anterior (x\u1D4F⁻¹).

        Retorno
        -------
        n : np.ndarray
            Retorna o valor da distância.
        """

        return np.linalg.norm(self.xk - self.all_iterations[self.k - 1])

    def algorithm(self) -> None:
        """O algoritmo do método escolhido, MG ou MGRP.

        Explicação
        ----------
        É calculado 100 iterações para cada caso de teste, a não ser que o critério de parada seja atingido.

        Critério de parada:
        - O resultado de `x\u1D4F` é igual ao valor esperado `x*`.
        - A distância entre a iteração atual (x\u1D4F) e o resultado da iteração anterior (x\u1D4F⁻¹) é menor ou igual a tolerância.
        """

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
