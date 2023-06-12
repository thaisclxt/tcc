from rich.console import Console
from rich.table import Table


def unpack_data(table, data):
    for row in zip(*data):
        table.add_row(*row)


def title(is_MG, function):
    title = '\nTestes do Método do Gradiente '
    title += '- MG' if is_MG else 'com Regularização Proximal - MGRP'
    title += f' para a função [bold]{function}[/bold]\n'
    return title


def generate_table(is_MG, function, lambda_row, tolerance_row, time_row, k_row, x_row, norm_row):
    table = Table(title=title(is_MG, function))

    table.add_column("nº", justify="right", style="white")
    table.add_column("tol", justify="right", style="cyan")

    if not is_MG:
        table.add_column("λk", justify="center", style="yellow")

    table.add_column("T\u0304", justify="center", style="blue")
    table.add_column("k\u0304", justify="right", style="magenta")
    table.add_column("x\u0304\u1D4F", justify="center", style="green")
    table.add_column("\u2225x\u0304\u1D4F - x*\u2225",
                     justify="center", style="red")

    rows = 4 if is_MG else 12
    for i in range(rows):
        if is_MG:
            table.add_row(
                str(i), tolerance_row[i], time_row[i], k_row[i], x_row[i], norm_row[i])
        else:
            table.add_row(str(
                i), tolerance_row[i], lambda_row[i], time_row[i], k_row[i], x_row[i], norm_row[i])

    console = Console()
    console.print(table)
