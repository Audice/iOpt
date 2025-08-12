from io import StringIO
import numpy as np


class DataDescription(object):
    """
    Описание областей определения и значения
    """

    @staticmethod
    def calculate_bounds_from_values(data):
        """
        Вычисление граничных значений в наборе табличных данных
        @param data: ndarray. Двумерный массив точек.
            Первое измерение массива это индекс точки,
            второе измерение массива это индекс координаты точки
        @returns: ndarray. Двумерный массив границ,
            первая строка - левая граница по координатам
            вторая строка - правая граница по координатам
        """
        data_dim = data.shape[1]
        data_min = np.zeros(data_dim)
        data_max = np.ones(data_dim)
        if data.size > 0:
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
        return np.vstack([data_min, data_max])

    def __init__(self, x_dim, y_dim, x_name=None, x_bounds=None, y_name=None, y_bounds=None):
        """
        Конструктор
        @param x_dim: размерность области определения
        @param y_dim: размерность области значений
        @param x_name: ndarray. Одномерный массив строковых имен параметров x
        @param x_bounds ndarray. Двумерный массив границ области определения,
            первая строка - левая граница по координатам
            вторая строка - правая граница по координатам
        @param y_name: ndarray. Одномерный массив строковых имен параметров y
        @param y_bounds: ndarray. Двумерный массив границ области значений,
            первая строка - левая граница по координатам
            вторая строка - правая граница по координатам
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_name = x_name
        self.x_bounds = x_bounds
        self.y_name = y_name
        self.y_bounds = y_bounds
        self.init()

    @staticmethod
    def from_config(model_config):
        x_dim = model_config['x_dim']
        y_dim = model_config['y_dim']
        x_name = None
        if 'x_name' in model_config:
            x_name = np.array(model_config['x_name'])
        y_name = None
        if 'y_name' in model_config:
            y_name = np.array(model_config['y_name'])
        x_bounds = None
        if 'x_bounds' in model_config:
            x_bounds = np.array(model_config['x_bounds']).T
        y_bounds = None
        if 'y_bounds' in model_config:
            y_bounds = np.array(model_config['y_bounds']).T
        return DataDescription(x_dim=x_dim, y_dim=y_dim,
                               x_name=x_name, y_name=y_name,
                               x_bounds=x_bounds, y_bounds=y_bounds)

    def to_config(self):
        return {
            'x_dim': self.x_dim,
            'y_dim': self.y_dim,
            'x_name': [name for name in self.x_name],
            'y_name': [name for name in self.y_name],
            'x_bounds': self.x_bounds.T.tolist(),
            'y_bounds': self.y_bounds.T.tolist()
        }

    def init(self):
        """
        Проверка и инициализация свойств объекта
        """
        if self.x_name is not None:
            assert self.x_dim == self.x_name.size
        if self.y_name is not None:
            assert self.y_dim == self.y_name.size
        if self.x_bounds is not None:
            assert 2 == self.x_bounds.ndim
            assert self.x_dim == self.x_bounds.shape[1]
        else:
            # создаем границы x по умолчанию [0; 1]
            self.x_bounds = np.vstack([np.zeros(self.x_dim), np.ones(self.x_dim)])
        if self.y_bounds is not None:
            assert 2 == self.y_bounds.ndim
            assert self.y_dim == self.y_bounds.shape[1]
        else:
            # создаем границы y по умолчанию [0; 1]
            self.y_bounds = np.vstack([np.zeros(self.y_dim), np.ones(self.y_dim)])
        if self.x_name is None:
            # создаем имена x по умолчанию
            self.x_name = np.array(['x{}'.format(i) for i in range(self.x_dim)])
        if self.y_name is None:
            # создаем имена y по умолчанию
            self.y_name = np.array(['y{}'.format(i) for i in range(self.y_dim)])

    def get_x_dimension(self) -> int:
        """
        Размерность области определения
        """
        return self.x_dim

    def get_x_bounds(self):
        """
        Границы области определения
        @return: ndarray. Двумерный массив границ,
            первая строка - левая граница по координатам
            вторая строка - правая граница по координатам
        """
        return self.x_bounds

    def get_x_index(self, name):
        """
        Возвращает индекс входного значения по имени
        @param name: str. Имя входного параметра
        @return int. Индекс параметра
        """
        return np.argmax(self.x_name == name)

    def get_y_index(self, name):
        """
        Возвращает индекс выходного значения по имени
        @param name: str. Имя выходного параметра
        @return int. Индекс параметра
        """
        return np.argmax(self.y_name == name)

    def set_x_bounds(self, name=None, index=None, min=None, max=None) -> None:
        """
        Установка границ области определения
        @param name: имя устанавливаемой границы (опционально)
        @param index: индекс устанавливаемой границы (опционально)
        @param min: минимальное значение границы
        @param max: максимальное значение границы
        """
        if index is not None:
            if min is not None:
                self.x_bounds[0][index] = min
            if max is not None:
                self.x_bounds[1][index] = max
            return
        if name is not None:
            if min is not None:
                self.x_bounds[0][self.x_name == name] = min
            if max is not None:
                self.x_bounds[1][self.x_name == name] = max
        else:
            if min is not None:
                self.x_bounds[0] = min
            if max is not None:
                self.x_bounds[1] = max

    def get_y_dimension(self) -> int:
        """
        @return int. Размерность области значений
        """
        return self.y_dim

    def get_y_bounds(self):
        """
        Границы области значений
        @return: ndarray. Двумерный массив границ,
            первая строка - левая граница по координатам
            вторая строка - правая граница по координатам
        """
        return self.y_bounds

    def set_y_bounds(self, name=None, index=None, min=None, max=None):
        """
        Установка границ области значений
        @param name: имя устанавливаемой границы (опционально)
        @param index: индекс устанавливаемой границы (опционально)
        @param min: минимальное значение границы
        @param max: максимальное значение границы
        """
        if index is not None:
            if min is not None:
                self.y_bounds[0][index] = min
            if max is not None:
                self.y_bounds[1][index] = max
            return
        if name is not None:
            if min is not None:
                self.y_bounds[0][self.x_name == name] = min
            if max is not None:
                self.y_bounds[1][self.x_name == name] = max
        else:
            if min is not None:
                self.y_bounds[0] = min
            if max is not None:
                self.y_bounds[1] = max


class Data(object):
    @staticmethod
    def get_norm(data, bounds):
        """
        Приведение координат точек в нормированный вид. Область значений точек заменяется на единичный гиперкуб
        @param data: ndarray. Двумерный массив точек.
            Первое измерение массива это индекс точки,
            второе измерение массива это индекс координаты точки
        @param bounds: ndarray. Двумерный массив границ,
            первая строка - левая граница по координатам
            вторая строка - правая граница по координатам
        @return: ndarray. Копия массива data с нормированными значениями
        """
        data_delta = bounds[1] - bounds[0]
        count_zero = len(data_delta) - np.count_nonzero(data_delta)
        if count_zero > 0:
            return data.copy()
        return (data.copy() - bounds[0]) / data_delta

    @staticmethod
    def get_denorm(data, bounds):
        """
        Приведение координат точек в денормированный вид.
        Область значений точек заменяется с единичного гиперкуба на заданную область
        @param data: ndarray. Двумерный массив точек.
            Первое измерение массива это индекс точки,
            второе измерение массива это индекс координаты точки
        @param bounds: ndarray. Двумерный массив границ,
            первая строка - левая граница по координатам
            вторая строка - правая граница по координатам
        @return: ndarray. Копия массива data с денормированными значениями
        """
        data_delta = np.float64(bounds[1] - bounds[0])
        # data_delta = bounds[1] - bounds[0]
        try:
            count_zero = data_delta.size - np.count_nonzero(data_delta)
        except Exception as inst:
            print(type(inst))  # the exception type
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
        if count_zero > 0:
            return data.copy()
        return data.copy() * data_delta + bounds[0]

    def __init__(self, x, y, description=None):
        """
        Конструктор
        @param x: ndarray. Двумерный массив точек из области определения.
            Первое измерение массива это индекс точки,
            второе измерение массива это индекс координаты точки
        @param y: ndarray. Двумерный массив точек из области значений.
            Первое измерение массива это индекс точки,
            второе измерение массива это индекс координаты точки
        @param description: DataDescription. Описание областей определения и значений (опционально)
        """
        self.x = x
        self.y = y
        self.description = description
        self.init()

    def init(self):
        """
        Проверка и инициализация свойств объекта
        """
        assert self.x.ndim == 2
        assert self.y.ndim == 2
        assert self.x.shape[0] == self.y.shape[0]
        if self.description is None:
            x_dim = self.x.shape[1]
            y_dim = self.y.shape[1]
            x_bounds = DataDescription.calculate_bounds_from_values(self.x)
            y_bounds = DataDescription.calculate_bounds_from_values(self.y)
            self.description = DataDescription(x_dim=x_dim, y_dim=y_dim, x_bounds=x_bounds, y_bounds=y_bounds)
        else:
            assert self.description.x_dim == self.x.shape[1]
            assert self.description.y_dim == self.y.shape[1]
            if ((self.description.y_bounds[0, :] == np.zeros((self.description.y_dim,))).all()
               and (self.description.y_bounds[1, :] == np.ones((self.description.y_dim,)))).all():
                self.description.y_bounds = np.array([np.min(self.y, axis=0).ravel(), np.max(self.y, axis=0).ravel()])
                if ((self.description.y_bounds[0, :] == self.description.y_bounds[1, :]).all()
                   and (self.description.y_bounds[1, :] == np.zeros((self.description.y_dim,))).all()):
                    self.description.y_bounds[0, :] = np.zeros((self.description.y_dim,))
                    self.description.y_bounds[1, :] = np.ones((self.description.y_dim,))

    def add(self, x, y):
        """
        Добавление новых данных
        @param x: Значения X
        @param y: Значения Y
        """
        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))

    def get_count(self) -> int:
        """
        @return: int. Число объектов данных
        """
        return self.x.shape[0]

    def get_description(self) -> DataDescription:
        """
        @return: DataDescription. Описание данных
        """
        return self.description

    def get_x_norm(self):
        """
        Получение нормированных данных из области опредления
        @return: ndarray. Двумерный массив нормированных значений x
        """
        return Data.get_norm(self.x, self.description.get_x_bounds())

    def get_y_norm(self):
        """
        Получение нормированных данных из области значений
        @return: ndarray. Двумерный массив нормированных значений y
        """
        return Data.get_norm(self.y, self.description.get_y_bounds())

    def get_x_denorm(self, x_norm):
        """
        Восстановление данных из нормированного состояния области определения
        @param x_norm: ndarray. Двумерный массив нормированных значений x
        @return: ndarray. Двумерный массив денормированных значений x
        """
        return Data.get_denorm(x_norm, self.description.get_x_bounds())

    def get_y_denorm(self, y_norm):
        """
        Восстановление данных из нормированного состояния области значений
        @param y_norm: ndarray. Двумерный массив нормированных значений y
        @return: ndarray. Двумерный массив денормированных значений y
        """
        return Data.get_denorm(y_norm, self.description.get_y_bounds())

    def to_csv_string(self):
        """
        Сохранение данных в строку формата CSV
        """
        xy = np.hstack([self.x, self.y])
        string_stream = StringIO()
        np.savetxt(string_stream, xy, delimiter=';')
        return string_stream.getvalue()

    def merge(self, other):
        """
        Присоединение другого набора данных к текущему. Оба набора должны иметь одинаковые размерности данных
        @param other: Data. Набор данных, который будет присоединен к текущему
        """
        assert self.description.x_dim == other.description.x_dim
        assert self.description.y_dim == other.description.y_dim
        self.x = np.vstack([self.x, other.x])
        self.y = np.vstack([self.y, other.y])
        self.x, indices = np.unique(self.x, axis=0, return_index=True)
        self.y = self.y[indices]

    @staticmethod
    def load_csv_file(file, data_description=None):
        """
        Загрузка данных из файла в формате CSV
        @param file: объект файла, из которого осуществляется чтение
        @param data_description: DataDescription. описание модели данных
        """
        xy = np.genfromtxt(file, delimiter=';')
        if xy.ndim < 2:
            xy = xy.reshape(0, -1)
        x_dim = xy.shape[1] - 1
        if data_description is not None:
            x_dim = data_description.x_dim
        x, y = np.split(xy, indices_or_sections=[x_dim], axis=1)
        return Data(x, y, data_description)
