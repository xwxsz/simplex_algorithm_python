import numpy as np

def simplex(c, A, b):
    """
    Реализация симплекс-метода решения задачи ЛП в канонической форме
    Входные данные:
    c - коэффиценты функции
    A - матрица ограничений
    b - вектор ограничений (правые части)
    Возвращает:
    solution - оптимальные значения x
    max_value - максимум функции
    """

    # Определяем количество переменных и ограничений
    num_vars = c.shape[0]
    num_constraints = A.shape[0]

    # Создаем пусткую симплекс-таблицу
    table = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

    # Заполняем симплекс-таблицу
    table[:num_constraints, :num_vars] = A
    table[:num_constraints, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    table[:num_constraints, -1] = b
    table[-1, :num_vars] = -c

    print("Начальная симплекс-таблица:")
    print(table)
    print("-" * 40)

    iteration = 0

    while True:
        print(f"\nИтерация {iteration}:")
        iteration += 1

        # Ищем разрешающий столбец
        pivot_col = np.argmin(table[-1, :-1])
        if table[-1, pivot_col] >= 0:
            print("Оптимум достигнут.")
            break  # Оптимум достигнут

        # Ищем разрешающую строку
        ratios = table[:-1, -1] / table[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)

        print(f"Входящая переменная: столбец {pivot_col}")
        print(f"Исходящая переменная: строка {pivot_row}")

        # Пивотирование
        table[pivot_row, :] /= table[pivot_row, pivot_col]
        for i in range(table.shape[0]):
            if i != pivot_row:
                table[i, :] -= table[i, pivot_col] * table[pivot_row, :]

        # Вывод промежуточной симплекс-таблицы
        print("Промежуточная симплекс-таблица:")
        print(table)
        print("-" * 40)

    # Оптимальное решение
    solution = np.zeros(num_vars)

    # Проверка на базис
    for i in range(num_constraints):
        for j in range(num_vars):
            if table[i, j] == 1 and np.all(table[:, j] == np.eye(num_constraints + 1)[:, i]):
                solution[j] = table[i, -1]

    max_value = table[-1, -1]
    return solution, max_value

def run_test1():
    print("f(x) = -x1 + x2\nx1 + x2 <= 1\nx1, x2 >= 0")
    c = np.array([-1, 1])
    A = np.array([[1, 1]])
    b = np.array([1])
    sol, max_val = simplex(c, A, b)
    print(f"Решение: {sol}, Максимум: {max_val}")

def run_test2():
    print("f(x) = x1\nx1 + x2 <= 1\nx1, x2 >= 0")
    c = np.array([1, 0])
    A = np.array([[1, 1]])
    b = np.array([1])
    sol, max_val = simplex(c, A, b)
    print(f"Решение: {sol}, Максимум: {max_val}")

def run_test3():
    print("f(x) = 3x1 + 4x2\n4x1 + x2 <= 8\n-x1 + x2 <= 3\nx1, x2 >= 0")
    c = np.array([3, 4])
    A = np.array([[4, 1], [-1, 1]])
    b = np.array([8, 3])
    sol, max_val = simplex(c, A, b)
    print(f"Решение: {sol}, Максимум: {max_val}")
