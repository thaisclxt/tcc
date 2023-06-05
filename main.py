import utils
import numpy as np
from sympy import parse_expr, Expr
from function import Function
from methods.MG import MG
from rich.progress import track
from table import generate_table


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
    time_row = []
    k_row = []
    x_row = []
    norm_row = []

    print()

    for i in range(4):
        tolerance: float = 10**-(i+2)
        result_list = []
        k_list = []
        time_list = []

        for j in track(range(100), description='Processando...'):
            x0 = np.random.uniform(low=0.0, high=10.0, size=2)

            mg = MG(func, x0, tolerance)
            mg.algorithm()

            time_list.append(mg.processing_time)
            result_list.append(mg.xk)
            k_list.append(mg.k)

        result_mean = calculate_result_mean(result_list)

        tolerance_row.append(str(tolerance))
        time_row.append(str(np.mean(time_list)))
        k_row.append(str(np.mean(k_list)))
        x_row.append(str(result_mean))
        norm_row.append(str(norm(result_mean, x_star)))

    generate_table(function, tolerance_row, time_row, k_row, x_row, norm_row)


if __name__ == "__main__":
    main()
