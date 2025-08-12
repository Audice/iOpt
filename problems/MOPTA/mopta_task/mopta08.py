import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np

from problems.MOPTA.mopta_task import file_utils
from problems.MOPTA.mopta_task.constrained_model import ConstrainedModel
from problems.MOPTA.mopta_task.constraint import Constraint
from problems.MOPTA.mopta_task.data import DataDescription


def const(points: np.ndarray):
    result = np.zeros((points.shape[0], 1))
    for i in range(points.shape[0]):
        x = (points[i, :]).ravel()
        broke = len(x[x > 0])
        g = np.sum(x)
        g_not_broken = np.sum((x[x < 0])) * -1
        result[i] = g if (g > 0.01 or broke == 0) else 0.01 / (1 + g_not_broken)
    return result


def number_of_completed(points: np.ndarray):
    result = np.zeros((points.shape[0], 1))
    for i in range(points.shape[0]):
        x = (points[i, :]).ravel()
        result[i] = len(x[x < 0])
    return result


def number_of_violations(points: np.ndarray):
    result = np.zeros((points.shape[0], 1))
    for i in range(points.shape[0]):
        x = (points[i, :]).ravel()
        result[i] = len(x[x > 0])
    return result


def summa(points: np.ndarray):
    result = np.zeros((points.shape[0], 1))
    for i in range(points.shape[0]):
        result[i] = np.array(np.sum(points[i, :])).ravel()
    return result


def default(points: np.ndarray):
    return points


class Mopta08(ConstrainedModel):
    """
    MOPTA08 из datadvance
    https://www.datadvance.ru/ru/blog/use-cases/pseven-solves-mopta08-from-automotive-industry.html

    Исполняемый файл для windows: https://baxus.papenmeier.io/troubleshooting.html#mopta08-executables
    mopta08_amd64.exe должен лежать в директории модели

    Постановка:
    1 минимизируемая целевая функция - масса
    124 переменные, приведенные к нормали [0,1]
    68 ограничений в виде неравенств gi (x) ≤ 0
    Ограничения строго нормализованы: 0.05 означает превышение требований на 5% и т.д.
    В качестве начальной точки выбрана допустимая точка со значением целевой ~251.07

    Цель:

    Количество вычислений ~ 15 x Количество переменных (1860)
    Полностью допустимое решение (без нарушений ограничений)
    Значение целевой функции ≤ 228 (не менее 80% от потенциального снижения)
    """

    def __init__(self, convolution_constraints="default"):
        super(Mopta08, self).__init__(DataDescription(x_dim=124,
                                                      y_dim=1,
                                                      x_bounds=np.array([[0] * 124,
                                                                         [1] * 124])))
        """
        @param convolution_constraints: str. Свертка ограничений ("default", "sum", "number_of_violations","number_of_completed" , "const")
        """

        mopta_exectutable = "mopta08_amd64.exe"

        self.mopta_full_path = os.path.join(
            Path(__file__).parent, mopta_exectutable
        )

        self.convolution_constraints = convolution_constraints
        self.len_constr = 68
        if self.convolution_constraints != 'default':
            self.len_constr = 1
        self.constraints = None


        self.x_data = []
        self.y_data = []
        self.launches = 0

    def get_name(self) -> str:
        return 'mopta08'

    def _run_executable(self, point: np.ndarray):
        """
        @param point: np.ndarray. Точка для вычисления критерия и ограничений в ней
        """

        index = self.counted_before(point)
        if index >= 0:
            return np.array(self.y_data[index][0]).ravel(), np.array(self.y_data[index][1:]).reshape(68, 1)

        with open("input.txt", "w+") as input_file:
            for _x in point:
                input_file.write(f"{_x}\n")

        self.launches += 1
        print('launch', self.launches)

        with subprocess.Popen(self.mopta_full_path, stdout=subprocess.PIPE) as popen:
            popen.wait()
            with open("output.txt", "r") as output_file:
                output = output_file.read().split("\n")

        output = [x.strip() for x in output]
        output = np.array([float(o) for o in output if len(o) > 0])

        self.x_data.append(point)
        self.y_data.append(output)

        file_utils.write_to_csv('mopta_x_data.csv', self.x_data)
        file_utils.write_to_csv('mopta_y_data.csv', self.y_data)

        return np.array(output[0]).ravel(), np.array(output[1:]).reshape(68, 1)

    def get_constraints_y(self) -> List[Constraint]:
        if self.constraints is None:
            self.constraints = [Constraint(fun=lambda p: self.constr(p)[:, i], ub=0.) for i in range(self.len_constr)]
        # return []
        return self.constraints

    def get_constraints(self) -> List[Constraint]:
        return []

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1, self.get_data_description().get_x_dimension())
        n = points.shape[0]
        y = np.zeros((n, self.description.y_dim))
        # print('evaluate', n)
        for i in range(n):
            result, constraints = self._run_executable(points[i].ravel())
            # y[i] = np.array(result)
            y[i] = np.array(result).ravel()
        return y

    def constr(self, points: np.ndarray) -> np.ndarray:
        # print('constr')
        points = points.reshape(-1, self.get_data_description().get_x_dimension())
        n = points.shape[0]
        c = np.zeros((n, 68))
        for i in range(n):
            result, constraints = self._run_executable(points[i].ravel())
            c[i] = np.array(constraints).ravel()
            # c[i] = np.array(constraints)
        if self.convolution_constraints == "default":
            return default(c)
        if self.convolution_constraints == "sum":
            return summa(c)
        if self.convolution_constraints == "number_of_violations":
            return number_of_violations(c)
        if self.convolution_constraints == "number_of_completed":
            return number_of_completed(c)
        if self.convolution_constraints == "const":
            return const(c)
        return c

    def counted_before(self, point):
        for i in range(len(self.x_data)):
            if np.array_equal(point, self.x_data[i]):
                return i
        return -1