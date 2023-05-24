import numpy as np

from sympy import parse_expr, sympify, Expr
from function import Function
from methods.MG import MG


example_functions = {
    '3*x^2 - 12*x + 3*y^2 - 6*y + 25': {
        'initial_iteration': [-1, -1],
        'global_minimum': [2, 1]
    },
    '5*x^2 + 8*y^2 + 2*x*y - 42*x - 102*y': {
        'initial_iteration': [4, 4],
        'global_minimum': [3, 6]
    },
    'x^2*y + x*y^2 - 96*x*y': {
        'initial_iteration': [2, 2],
        'global_minimum': [32, 32]
    },
    'x^2 + 2*y^2 - 18*x - 24*y + 2*x*y + 120': {
        'initial_iteration': [9, 6],
        'global_minimum':  [6, 3]
    }
}

number_of_examples = len(example_functions)


def function_type() -> bool:
    print('Qual função deseja escolher?')
    print('1- Digitar uma função')
    print('2- Escolher uma função de exemplo')

    is_user_input = input() == '1'

    return is_user_input


def define_function(is_user_input: bool) -> str:
    if is_user_input:
        function = input('\nEntre com a função ex: 2x^2 + 2*y -32: ')
    else:
        print('\nQual dessas funções deseja escolher?')

        i = 0
        for function in example_functions.keys():
            print(f'{i}- {function}')
            i += 1

        choice = int(input())
        index = choice if choice < number_of_examples else number_of_examples - 1

        function = list(example_functions.keys())[index]

    return function


def define_global_minimun(is_user_input: bool, function: str) -> np.ndarray:
    if is_user_input:
        value = input('\nEntre com o ponto de mínimo global ex: (-1, -1): ')
        x_star = sympify(value)
    else:
        x_star = example_functions[function]['global_minimum']

    return np.array(x_star, dtype=float)


def main():
    is_user_input = function_type()
    function = define_function(is_user_input)
    x_star = define_global_minimun(is_user_input, function)

    expression: Expr = parse_expr(function.replace("^", "**"))

    func = Function(is_user_input, expression, x_star)
    print(f'\nO gradiente da função problema é: {func.gradient}')

    for i in range(4):
        tolerance: float = 10 ** (i + 2)

        for i in range(100):
            x0 = np.random.uniform(low=-10.0, high=10.0, size=2)
            print(f'\nA iteração inicial é: {x0}')

            mg = MG(func, x0, tolerance)
            mg.algorithm()


if __name__ == "__main__":
    main()
