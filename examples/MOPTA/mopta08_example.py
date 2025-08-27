from problems.MOPTA.mopta08 import Mopta08Problem

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

import numpy as np
from iOpt.trial import Point
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Решение задачи mopta08
    """

    problem = Mopta08Problem()

    x0 = np.loadtxt(
        'start_point.csv',
        delimiter=';', dtype=np.float64)

    # Формируем параметры решателя
    params = SolverParameters(r=3, eps=0.1, refine_solution=True, proportion_of_global_iterations=0.001, iters_limit=5000,  start_point=Point(x0))

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='result')
    solver.add_listener(cfol)

    # Решение задачи
    sol = solver.solve()

    val = problem.calculate(sol.best_trials[0].point, sol.best_trials[0].function_values[3])
