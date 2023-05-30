# import pandas as pd


# def generate_table(function, tolerance_row, time_row, k_row, x_row, norm_row):
#     table = pd.DataFrame({
#         "tol": tolerance_row,
#         "T\u0304": time_row,
#         "k\u0304": k_row,
#         "x\u0304\u1D4F": x_row,
#         "\u2225x\u0304\u1D4F - x*\u2225": norm_row
#     })

#     print(
#         f'\nTabela 1 - Testes do Método do Gradiente para a função {function}\n')

#     print(table)


from rich.console import Console
from rich.table import Table


def unpack_data(table, data):
    for row in zip(*data):
        table.add_row(*row)


def generate_table(function, tolerance_row, time_row, k_row, x_row, norm_row):
    title = f'\nTestes do Método do Gradiente para a função [bold]{function}[/bold]\n'
    table = Table(title=title)

    table.add_column("tol", justify="right", style="cyan")
    table.add_column("T\u0304", justify="center", style="blue")
    table.add_column("k\u0304", justify="right", style="magenta")
    table.add_column("x\u0304\u1D4F", justify="center", style="green")
    table.add_column("\u2225x\u0304\u1D4F - x*\u2225",
                     justify="center", style="red")

    for i in range(4):
        table.add_row(tolerance_row[i], time_row[i],
                      k_row[i], x_row[i], norm_row[i])

    console = Console()
    console.print(table)
