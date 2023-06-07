import time
import numpy as np

from sympy import simplify
from function import Function
from methods.method import Method
from rich.console import Console
from rich.table import Table


class MGRP(Method):
    def __init__(self, function: Function, tolerance: float, aa: int):
        super().__init__(function, tolerance)
        self.lamda: float = aa + 1 + (1 / (self.k+1))

    def arg_min(self, _gradient_xk):
        f = self.xk - np.array([self.alpha, self.alpha]) * _gradient_xk
        substitution = self.function.expression.subs(
            {self.x: f[0], self.y: f[1]})
        simp = simplify(substitution)
        mgrp = simp + self.alpha ** 2 * self.lamda * \
            (np.linalg.norm(np.array(_gradient_xk)) ** 2)
        return mgrp

    def unpack_data(table, data):
        for row in zip(*data):
            table.add_row(*row)

    def generate_table(self, lambda_row, tolerance_row, time_row, k_row, x_row, norm_row):
        title = f'\nTestes do Método do Gradiente com Regularização Proximal - MGRP para a função [bold]{self.function}[/bold]\n'
        table = Table(title=title)

        table.add_column("tol", justify="right", style="cyan")
        table.add_column("lamda", justify="right", style="white")
        table.add_column("T\u0304", justify="center", style="blue")
        table.add_column("k\u0304", justify="right", style="magenta")
        table.add_column("x\u0304\u1D4F", justify="center", style="green")
        table.add_column("\u2225x\u0304\u1D4F - x*\u2225",
                         justify="center", style="red")

        for _ in range(3):
            for i in range(4):
                table.add_row(tolerance_row[i], lambda_row[i], time_row[i],
                              k_row[i], x_row[i], norm_row[i])

        console = Console()
        console.print(table)

    def algorithm(self) -> None:
        start_time = time.time()

        while self.k < 100:
            if self.found_result():
                self.set_time(start_time)
                break

            _gradient_xk = self.gradient_xk()
            _arg_min = self.arg_min(_gradient_xk)
            _alpha_k = self.alpha_k(_arg_min)

            self.xk = self.result(_alpha_k)
            self.all_iterations.append(self.xk)

            self.k += 1

            if self.norm() <= self.tolerance:
                self.set_time(start_time)
                break
