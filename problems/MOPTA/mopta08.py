import numpy as np

from problems.MOPTA.mopta_task import mopta08

from iOpt.trial import FunctionType
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem


class Mopta08Problem(Problem):
    def __init__(self):
        """
        Mopta08 problem class constructor
        """
        super(Mopta08Problem, self).__init__()
        self.name = "Mopta08"
        self.dimension: int = 124
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 68

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        for i in range(self.dimension):
            self.lower_bound_of_float_variables[i] = 0

        self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        for i in range(self.dimension):
            self.upper_bound_of_float_variables[i] = 1
            
        self.mopta = mopta08.Mopta08()


        #self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        #pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        #pointfv = [0.941176, 0.941176]
        #KOpoint = Point(pointfv, [])
        #KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        #KOfunV[0] = FunctionValue()
        #KOfunV[0].value = -1.489444
        #self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculating the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated. 
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        res: np.double = 0
        x: np.double = point.float_variables
        x_np = np.array(x)

        if function_value.type == FunctionType.OBJECTIV:
            res = self.mopta.evaluate(x_np).item()
        else:
            res = self.mopta.constr(x_np).ravel()[function_value.functionID]# constraint #functionID

        function_value.value = res
        return function_value

    def get_name(self):
        return self.name



# class TestMopta08(TestCase):
#     def test_cobyla(self):
#         import numpy as np
#         from scipy.optimize import minimize
#         import matplotlib.pyplot as plt

#         mopta = Mopta08()


#         # Тестовая целевая функция
#         def obj(x):
#             return mopta.evaluate(x)

#         # Тестовая функция ограничений
#         def constr(x):
#             return mopta.constr(x)

#         # Хранение значений целевой функции для отслеживания сходимости
#         obj_values = []

#         # Обертка для целевой функции, чтобы сохранять значения
#         def obj_with_history(x):
#             value = obj(x)
#             obj_values.append(value.ravel())
#             return value

#         # Ограничения для COBYLA (должны быть >= 0)
#         def constraints_fun(x):
#             return -constr(x).ravel() # Отрицаем, так как COBYLA требует >= 0

#         # Начальное приближение
#         x0 = np.loadtxt(
#             'start_point.csv',
#             delimiter=';', dtype=np.float64)

#         # Настройка ограничений для COBYLA
#         constraints = [
#             {'type': 'ineq', 'fun': lambda x: constraints_fun(x)},
#             {'type': 'ineq', 'fun': lambda x: x },  # x_i ≥ 0
#             {'type': 'ineq', 'fun': lambda x: 1.0 - x}  # x_i ≤ 1
#         ]

#         # Запуск оптимизации
#         result = minimize(
#             fun=obj_with_history,
#             x0=x0,
#             method='COBYLA',
#             constraints=constraints,
#             options={'maxiter': 300, 'disp': True}
#         )

#         # Визуализация результатов
#         # 1. График сходимости целевой функции
#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.plot(obj_values, label='Целевая функция')
#         plt.xlabel('Итерация')
#         plt.ylabel('Значение целевой функции')
#         plt.title('Сходимость')
#         plt.grid(True)
#         plt.legend()

#         # 2. Гистограмма значений переменных
#         plt.subplot(1, 3, 2)
#         plt.hist(result.x, bins=20, edgecolor='black')
#         plt.xlabel('Значение переменной')
#         plt.ylabel('Частота')
#         plt.title('Распределение переменных')
#         plt.grid(True)

#         # 3. График значений ограничений
#         constraint_values = constr(result.x).ravel()
#         plt.subplot(1, 3, 3)
#         plt.bar(range(68), constraint_values)
#         plt.axhline(y=0, color='r', linestyle='--', label='Порог (<= 0)')
#         plt.xlabel('Номер ограничения')
#         plt.ylabel('Значение ограничения')
#         plt.title('Значения ограничений')
#         plt.grid(True)
#         plt.legend()

#         plt.tight_layout()
#         plt.show()

#         # Вывод результатов
#         print("Статус оптимизации:", result.message)
#         print("Значение целевой функции:", result.fun)
#         print("Найденное решение (первые 10 элементов):", result.x[:10])
#         print("Количество оценок функции:", result.nfev)
#         print("Максимальное нарушение ограничений:", np.max(constr(result.x)))
