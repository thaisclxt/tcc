import time
import utils

import numpy as np
import matplotlib.pyplot as plt

from table import generate_table
from sympy import parse_expr, Expr
from function import Function
from method import Method
from methods.MGRP import MGRP
from rich.progress import track


def main():
    is_MG = utils.method_type()
    is_user_input = utils.function_type()
    function = utils.define_function(is_user_input)
    x_star = utils.define_global_minimun(is_user_input, function)

    expression: Expr = parse_expr(function.replace("^", "**"))

    func = Function(is_user_input, expression, x_star)

    tolerance_row = []
    lambda_row = []
    time_row = []
    k_row = []
    x_row = []
    norm_row = []

    print()

    scatter_legend = []

    extern_range = 1 if is_MG else 3
    table_rows = 4
    total_tests = 100
    colors = ["red", "orange", "yellow", "green", "blue", "magenta",
              "cyan", "purple", "brown", "gray", "black", "pink"]

    start_time = time.time()

    for i in range(extern_range):
        for j in range(table_rows):
            tolerance: float = 10**-(j+2)
            result_list = []
            k_list = []
            time_list = []

            for _ in track(range(total_tests), description="Processando..."):
                method = Method(is_MG, func, tolerance, i)
                method.algorithm()

                time_list.append(method.processing_time)
                result_list.append(method.xk)
                k_list.append(method.k)

            result_mean = utils.calculate_result_mean(result_list)

            index = j + table_rows * i

            plt.scatter(*zip(*result_list), color=colors[index])
            scatter_legend.append(f'Caso de teste {index}')

            tolerance_row.append(str(tolerance))
            time_row.append(str(np.mean(time_list)))
            k_row.append(str(np.mean(k_list)))
            x_row.append(str(result_mean))
            norm_row.append(str(utils.norm(result_mean, x_star)))

            if not is_MG:
                if i == 0:
                    lambda_row.append('1 / (k+1)')
                elif i == 1:
                    lambda_row.append('1 + (1 / (k+1))')
                else:
                    lambda_row.append('2')

        print()

    generate_table(is_MG, function, lambda_row, tolerance_row, time_row, k_row, x_row, norm_row)

    end_time = int(time.time() - start_time)
    print(f'\nTempo de processamento total: {end_time // 60}min {end_time % 60}sec')

    plt.legend(scatter_legend)
    plt.show()


if __name__ == "__main__":
    main()
