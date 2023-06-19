import numpy as np

from sympy import sympify


# Funções de exemplo usadas no TCC 1
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


# A quantidade de funções de exemplo
number_of_examples = len(example_functions)


def method_type() -> bool:
    """Define o tipo de método a ser computado.

    Explicação
    ----------
    Pode ser o Método do Gradiente - MG ou o Método do Gradiente com Regularização Proximal - MGRP.

    Retorno
    -------
    Se for MG, retorna `True`, se não retorna `False`.
    """

    print('Por qual método deseja executar o algorítmo?')
    print('1- MG - Método do Gradiente')
    print('2- MGRP - Método do Gradiente com Regularização Proximal')

    return input() == '1'


def function_type() -> bool:
    """Define o tipo da função escolhida.

    Explicação
    ----------
    Pode ser uma função de entrada ou uma função de exemplo usada no TCC 1.

    Retorno
    -------
    Se for o 1º caso, retorna `True`, se não retorna `False`.
    """

    print('\nQual função deseja escolher?')
    print('1- Digitar uma função')
    print('2- Escolher uma função de exemplo')

    return input() == '1'


def define_function(is_user_input: bool) -> str:
    """Define qual é a função escolhida.

    Explicação
    ----------
    Caso o usuário deseja uma função usada no TCC 1, ele terá que escolher qual das funções de `example_functions` usar.

    Caso contrário, basta entrar com a função desejada.

    Parâmetro
    ---------
    is_user_input : bool
        Variável que valida se a função é do tipo "entrada do usuário".

    Retorno
    -------
    Será retornado a função desejada no tipo `str`.
    """

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
    """Define o ponto de mínimo global `x*` da função escolhida.

    Explicação
    ----------
    Caso o usuário tenha escolhido uma função usada no TCC 1, `x*` já foi definido em `example_functions`.

    Caso contrário, ele terá que entrar com o valor de `x*`.

    Parâmetro
    ---------
    is_user_input : bool
        Variável que valida se a função é do tipo "entrada do usuário".
    function : str
        A função escolhida.

    Retorno
    -------
    Será retornado um vetor do tipo `np.ndarray` com o valor de `x*`.
    """

    if is_user_input:
        value = input('\nEntre com o ponto de mínimo global ex: (-1, -1): ')
        x_star = sympify(value)
    else:
        x_star = example_functions[function]['global_minimum']

    return np.array(x_star, dtype=float)


def calculate_result_mean(list: list[np.ndarray]) -> tuple:
    """Calcula a média de todos os `x\u1D4F` -> `x\u0304\u1D4F`.

    Parâmetro
    ---------
    list : list[np.ndarray]
        Uma lista de todos os resultados de `x\u1D4F`.

    Retorno
    -------
    Retorna um objeto com dois valores do tipo `float`, que são referentes ao cálculo da média de x e de y.
    """

    media_x: float = np.mean(np.array(list)[:, 0])
    media_y: float = np.mean(np.array(list)[:, 1])

    return media_x, media_y


def norm(a: tuple, b: np.ndarray) -> np.ndarray:
    """Calcula \u2225x\u0304\u1D4F - x*\u2225.

    Parâmetros
    ---------
    a : tuple
        A média de todos os `x\u1D4F`.
    b : np.ndarray
        O ponto de mínimo global `x*`.

    Retorno
    -------
    n : `np.ndarray`
        A distância entre os dois vetores `a` e `b`.
    """

    return np.linalg.norm(a - b)
