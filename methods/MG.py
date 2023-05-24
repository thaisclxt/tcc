import numpy as np

from function import Function


class MG():
    def __init__(self, function: Function, initial_iteration: np.ndarray, tolerance: float):
        self.function = function
        self.x0 = initial_iteration
        self.tolerance = tolerance
        self.k = 0
        self.xk_list: list[np.ndarray] = [self.x0]
        self.time = self.calculate_time
        self.result = self.response

    def calculate_time():
        pass

    def response():
        pass

    def gradient_xk(self) -> np.ndarray:
        gradient = self.function.gradient
        xk = self.xk_list[self.k]

        a = gradient[0].subs({self.function.x: xk[0], self.function.y: xk[1]})
        b = gradient[1].subs({self.function.x: xk[0], self.function.y: xk[1]})

        return np.array([a, b], dtype=float)

    def norm(self) -> float:
        return np.linalg.norm(self.xk_list[self.k] - self.xk_list[self.k - 1])

    def algorithm(self):
        # Laço de repetição que será executado enquanto k for menor que 100, para fazer 100 casos de teste, a não ser que o critério de parada seja alcançado
        while self.k < 100:
            # Verifica se o resultado encontrado é igual ao resultado experado
            if np.array_equal(self.xk_list[self.k], self.function.global_minimun):
                break

            _gradient_xk = self.gradient_xk()

            # self.k += 1

            # Verifica se a distância entre a iteração atual e a iteração anterior é menor ou igual à tolerância
            if self.norm() <= self.tolerance:
                break
