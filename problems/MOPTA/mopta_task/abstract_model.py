import numpy as np


from problems.MOPTA.mopta_task.data import DataDescription

class AbstractModel(object):
    """
    Интерфейс объекта, являющегося целью параметрических исследований и оптимизации
    """

    def __init__(self, description: DataDescription):
        self.description = description

    @staticmethod
    def get_name() -> str:
        """
        Возвращает имя модели, может быть переопределено у наследников
        """
        raise NotImplementedError

    def get_data_description(self) -> DataDescription:
        """
        Возвращает описание областей определения и значения
        """
        return self.description

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Вычисление значения функции в точке
        @param points: ndarray. Двумерный массив точек.
            Первое измерение массива это индекс точки,
            второе измерение массива это индекс координаты точки
        @return: ndarray. Двумерный массив значений модели в поданных точках.
            Первое измерение массива это индекс точки,
            второе измерение массива это индекс координаты точки
        """
        raise NotImplementedError
