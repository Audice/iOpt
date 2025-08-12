import sys


class Constraint(object):
    """
    Класс ограничения
    """

    def __init__(self, fun: callable,
                 lb: float = -sys.float_info.max,
                 ub: float = sys.float_info.max):
        """
        @fun: callable. Функция ограничения
        @lb: float. Нижняя граница
        @ub: float. Верхняя граница
        """
        self.fun = fun
        self.lb = lb
        self.ub = ub
