from sympy import sympify
import numpy as np


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


def method_type() -> bool:
    print('Por qual método deseja executar o algorítmo?')
    print('1- MG - Método do Gradiente')
    print('2- MGRP - Método do Gradiente com Regularização Proximal')

    return input() == '1'


def function_type() -> bool:
    print('\nQual função deseja escolher?')
    print('1- Digitar uma função')
    print('2- Escolher uma função de exemplo')

    return input() == '1'


def define_function(is_user_input: bool) -> str:
    if is_user_input:
        return input('\nEntre com a função ex: 2x^2 + 2*y -32: ')
    else:
        print('\nQual dessas funções deseja escolher?')

        i = 0
        for function in example_functions.keys():
            print(f'{i}- {function}')
            i += 1

        choice = int(input())
        index = choice if choice < number_of_examples else number_of_examples - 1

        return list(example_functions.keys())[index]


def define_global_minimun(is_user_input: bool, function: str) -> np.ndarray:
    if is_user_input:
        value = input('\nEntre com o ponto de mínimo global ex: (-1, -1): ')
        x_star = sympify(value)
    else:
        x_star = example_functions[function]['global_minimum']

    return np.array(x_star, dtype=float)
