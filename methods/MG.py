import numpy as np

from scipy import optimize
from sympy import simplify, lambdify
from function import Function


class MG():
    def __init__(self, function: Function, initial_iteration: np.ndarray, tolerance: float):
        self.function = function
        self.x0 = initial_iteration
        self.tolerance = tolerance

        self.k = 0
        self.xk_list: list[np.ndarray] = [self.x0]
        self.xk = self.xk_list[self.k]

        self.x = self.function.x
        self.y = self.function.y
        self.alpha = self.function.alpha

        self.time = self.calculate_time

    def calculate_time():
        pass

    # TODO: Verificar se o gradiente de xk está sendo executado mais de uma vez o que pode ser feito para melhorar
    def result(self) -> np.ndarray:
        alpha_k = self.alpha_k()
        return self.xk - np.array([alpha_k, alpha_k]) * self.gradient_xk()

    def alpha_k(self):
        f = lambdify(self.alpha, self.arg_min())

        # TODO: verificar os resultados, pois estão pais para os casos que não são o primeiro
        # TODO: retirar esses warnings que estão imprimindo
        resultado = optimize.fmin_bfgs(f, 0)
        return resultado[0]

    def arg_min(self):
        f = self.xk - np.array([self.alpha, self.alpha]) * self.gradient_xk()
        return simplify(self.function.expression.subs({self.x: f[0], self.y: f[1]}))

    def gradient_xk(self) -> np.ndarray:
        gradient = self.function.gradient

        a = gradient[0].subs({self.x: self.xk[0], self.y: self.xk[1]})
        b = gradient[1].subs({self.x: self.xk[0], self.y: self.xk[1]})

        return np.array([a, b], dtype=float)

    def norm(self) -> float:
        return np.linalg.norm(self.xk_list[self.k] - self.xk_list[self.k - 1])

    def algorithm(self):
        # Laço de repetição que será executado enquanto k for menor que 100, para fazer 100 casos de teste, a não ser que o critério de parada seja alcançado
        while self.k < 100:
            # Verifica se o resultado encontrado é igual ao resultado experado
            if np.array_equal(self.xk_list[self.k], self.function.global_minimun):
                break

            self.response = self.result()

            # TODO: parece que ta adicionando o ultimo valor duplicado
            self.xk_list.append(self.response)

            print(f'Iteração {self.k} = {self.xk_list[self.k]}')

            self.k += 1

            # TODO: não posso calcular a norma da x0 - a x1
            # Verifica se a distância entre a iteração atual e a iteração anterior é menor ou igual à tolerância
            if self.norm() <= self.tolerance:
                break
