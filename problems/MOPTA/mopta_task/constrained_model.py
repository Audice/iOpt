from typing import List

from problems.MOPTA.mopta_task.abstract_model import AbstractModel
from problems.MOPTA.mopta_task.constraint import Constraint


class ConstrainedModel(AbstractModel):
    """
    Модель с ограничениями
    """

    @staticmethod
    def get_name() -> str:
        return "constrained_model"

    def get_constraints(self) -> List[Constraint]:
        """
        Возвращает список ограничений
        """
        return []

    def get_constraints_y(self) -> List[Constraint]:
        """
        Возвращает список ограничений
        """
        return []