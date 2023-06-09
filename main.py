import time
import utils
import numpy as np

from sympy import parse_expr, Expr
from function import Function
from methods.MG import MG
from methods.MGRP import MGRP
from rich.progress import track


def calculate_result_mean(list: list[np.ndarray]):
    media_x = np.mean(np.array(list)[:, 0])
    media_y = np.mean(np.array(list)[:, 1])

    return media_x, media_y


def norm(a, b):
    return np.linalg.norm(a - b)


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

    extern_range = 1 if is_MG else 3
    table_rows = 4
    total_tests = 1

    start_time = time.time()

    for index in range(extern_range):
        for row in range(table_rows):
            tolerance: float = 10**-(row+2)
            result_list = []
            k_list = []
            time_list = []

            for _ in track(range(total_tests), description="Processando..."):
                method = MG(func, tolerance) if is_MG else MGRP(func, tolerance, index)
                method.algorithm()

                time_list.append(method.processing_time)
                result_list.append(method.xk)
                k_list.append(method.k)

            result_mean = calculate_result_mean(result_list)

            tolerance_row.append(str(tolerance))
            time_row.append(str(np.mean(time_list)))
            k_row.append(str(np.mean(k_list)))
            x_row.append(str(result_mean))
            norm_row.append(str(norm(result_mean, x_star)))

            if not is_MG:
                lambda_row.append(f'{index} + 1 + (1 / (k+1))')
        
        print()

    if is_MG:
        method.generate_table(tolerance_row, time_row, k_row, x_row, norm_row)
    else:
        method.generate_table(lambda_row, tolerance_row, time_row, k_row, x_row, norm_row)

    end_time = int(time.time() - start_time)
    print(f'\nTempo de processamento total: {end_time // 60}min {end_time % 60}sec')

if __name__ == "__main__":
    main()
