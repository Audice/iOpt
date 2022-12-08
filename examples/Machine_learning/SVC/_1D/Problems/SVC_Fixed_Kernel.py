import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from sko.GA import GA_TSP
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict

class SVC_Fixed_Kernel(Problem):

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray, kernel_coefficient_value: float,
                 regularization_bound: Dict[str, float]):
        self.dimension = 1
        self.numberOfFloatVariables = 1
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        self.kernelCoefficient = kernel_coefficient_value
        self.floatVariableNames = np.array(["Regularization parameter"], dtype=str)
        self.lowerBoundOfFloatVariables = np.array([regularization_bound['low']], dtype=np.double)
        self.upperBoundOfFloatVariables = np.array([regularization_bound['up']], dtype=np.double)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        cs = point.floatVariables[0]
        clf = SVC(C=10**cs, gamma=10**self.kernelCoefficient)
        clf.fit(self.x, self.y)
        functionValue.value = -cross_val_score(clf, self.x, self.y,
                                              scoring=lambda model, x, y: f1_score(y, model.predict(x))).mean()
        return functionValue
