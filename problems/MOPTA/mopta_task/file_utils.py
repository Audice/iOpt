import os
import json
import shutil
import tarfile
import tempfile
import time
import zipfile
import cProfile
from io import BytesIO

import numpy as np
import jsonschema


def exists(filepath):
    """проверка существования файла"""
    return os.path.exists(filepath)


def read_from_csv(filepath, delim=';', absolute_path=True):
    """чтение табличных числовых данных в двумерный массив"""
    if not absolute_path:
        filepath = get_absolute_path(filepath)
    result = np.genfromtxt(filepath, delimiter=delim)
    if result.size == 0:
        result = result.reshape(0, 0)
    return result


def write_to_csv(filepath, data, delim=';', absolute_path=False):
    """запись табличных числовых данных в файл"""
    if not absolute_path:
        filepath = get_absolute_path(filepath)
    np.savetxt(filepath, data, delimiter=delim)


def write_string_to_file(filepath, value):
    """
    запись строки в текстовый файл
    """
    with open(filepath, 'w') as fh:
        fh.write(value)


def write_strings_to_file(filepath, value):
    """
    запись строки в текстовый файл
    """
    with open(filepath, 'w') as fh:
        for i in range(len(value)):
            fh.write(value[i] + '\n')


def get_absolute_path(relative_path, if_full_path=False):
    relative_path = relative_path.replace('\\\\', '/').replace('\\', '/')
    if if_full_path:
        return relative_path
    """вычисление абсолютного пути файла относительно корня проекта"""
    return os.path.join(os.path.dirname(__file__), '../../' + relative_path).replace('\\\\', '/').replace('\\', '/')


def open_valid_json(json_file, schema_name, is_full_path=True):
    """
    Открытие и валидация JSON объекта из файла, согласно схеме
    """
    if not is_full_path:
        json_file = get_absolute_path(json_file)
    with open(json_file, 'r') as fh:
        return check_valid_json(fh.read(), schema_name)


def write_json(dictionary, filepath='result.json'):
    """
    Запись словаря в JSON файл
    """
    with open(filepath, 'w') as fp:
        json.dump(dictionary, fp, indent='\t')


def check_valid_json(json_string, schema_name):
    """
    Валидация JSON объекта из строки, согласно схеме
    """
    schema_file_path = get_absolute_path('schemas/' + schema_name)
    with open(schema_file_path, 'r') as fh:
        schema_file_path = schema_file_path.replace('\\', '/')
        schema = json.loads(fh.read().replace('${file_path}', schema_file_path))
    json_object = json.loads(json_string)
    jsonschema.validate(instance=json_object, schema=schema)
    return json_object


def get_line_count(file, count_empty=False):
    result = 0
    with open(file, 'r') as fh:
        for line in fh:
            if count_empty or line.strip():
                result += 1
    return result


def write_file_to_archive(archive_path, file_name, file_content):
    """
    Добавление текстового файла в архив. Если архив не существует, то он будет создан
    :param archive_path: абсолютный путь к архиву
    :param file_name: путь файла в архиве, без лидирующего слеша
    :param file_content: String. Строка с содержимым файла
    """
    mode = 'w'
    if os.path.exists(archive_path):
        mode = 'a'
    with zipfile.ZipFile(archive_path, mode, compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(file_name, file_content)


def write_file_to_archive_tar(archive_path, file_name, file_content, mode='a'):
    with tarfile.open(archive_path, mode) as tar:
        # Преобразуем содержимое в bytes, если это строка
        if isinstance(file_content, str):
            file_content = file_content.encode('utf-8')

        # Создаем объект TarInfo с метаданными файла
        tarinfo = tarfile.TarInfo(name=file_name)
        tarinfo.size = len(file_content)
        tarinfo.mtime = time.time()
        tarinfo.mode = 0o644  # Права доступа rw-r--r--

        # Добавляем файл в архив
        tar.addfile(tarinfo, BytesIO(file_content))


def fix_broken_encoding(broken_text):
    """Исправляет строку, где UTF-8 был ошибочно прочитан как CP1251"""
    try:
        return broken_text.encode('cp1251').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return broken_text  # Возвращаем как есть, если не получается исправить


def decode_with_fallback(data, encodings=('utf-8', 'cp1251', 'latin1')):
    """Пробует декодировать данные разными кодировками"""
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')  # В крайнем случае, с заменой ошибок


def extract_fix_and_update_json(archive_path, file_in_archive):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_archive.tar')

    try:
        with tarfile.open(archive_path, 'r') as tar_in, \
                tarfile.open(temp_path, 'w', encoding='utf-8') as tar_out:

            for member in tar_in.getmembers():
                if member.isfile():
                    content = tar_in.extractfile(member).read()

                    if member.name == file_in_archive:
                        # Для JSON-файла пробуем разные кодировки
                        text_content = decode_with_fallback(content)

                        try:
                            # Пробуем загрузить JSON
                            json_data = json.loads(text_content)

                            # Рекурсивно исправляем кодировку во всем JSON
                            def fix_encoding(obj):
                                if isinstance(obj, str):
                                    return fix_broken_encoding(obj)
                                elif isinstance(obj, dict):
                                    return {k: fix_encoding(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [fix_encoding(item) for item in obj]
                                return obj

                            fixed_data = fix_encoding(json_data)
                            fixed_content = json.dumps(fixed_data, ensure_ascii=False, indent=2).encode('utf-8')
                        except json.JSONDecodeError:
                            fixed_content = content  # Если не JSON, оставляем как есть
                    else:
                        fixed_content = content  # Для не-JSON файлов

                    # Создаем новый член архива
                    new_member = tarfile.TarInfo(name=member.name)
                    new_member.size = len(fixed_content)
                    new_member.mtime = member.mtime
                    new_member.mode = member.mode
                    tar_out.addfile(new_member, BytesIO(fixed_content))
                else:
                    tar_out.addfile(member)  # Директории и симлинки

        # Заменяем оригинальный архив (Windows требует особого подхода)
        shutil.move(temp_path, archive_path)
        return True

    except Exception as e:
        raise RuntimeError(f"Failed to update archive: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def write_file_to_archive_2(archive_path, file_name):
    with zipfile.ZipFile(archive_path, mode="a") as archive:
        archive.write(file_name)


def read_file_from_archive_as_string(archive_path, file_name):
    """
    Чтение содержимого файла, находящегося в архиве, в виде строки
    :param archive_path: str. Абсолютный путь к архиву
    :param file_name: str. Относительный путь файла в архиве
    :returns: str. Текстовое содерижмое файла
    """
    with zipfile.ZipFile(archive_path, 'r') as archive:
        file = archive.read(file_name)
        return file.decode('utf-8')


def read_file_from_archive(archive_path, file_name):
    """
    Открытие файла, находящегося в архиве
    :param archive_path: str. Абсолютный путь к архиву
    :param file_name: str. Относительный путь файла в архиве
    :returns: file. Файл, доступный для чтения
    """
    return zipfile.ZipFile(archive_path, 'r').open(file_name)


def read_model_from_csv(filename, x_dim, absolute_path=False):
    """
    Чтение числовых данных модели из файла
    :param filename: str. Имя файла в preCalcData.
    :param x_dim: int. Количество входных параметров модели.
    """
    data = read_from_csv(filepath=filename, delim=';', absolute_path=absolute_path)
    data = np.delete(data, 0, axis=0)  # удаляем заголовки
    data = data[:, 2:]  # удаляем первые 2 столбца индексов и статуса
    x, y = np.split(data, indices_or_sections=[x_dim], axis=1)  # разделяем данные на x и y
    return x, y


def read_model_by_unique_index_from_csv(filename, x_index, x_dim=None, y_index=None, absolute_path=False):
    """
    Чтение числовых данных модели из файла
    :param filename: str. Имя файла в preCalcData.
    :param x_dim: int. Количество входных параметров модели.
    """
    # приводим название индексов к массивам
    x_index = np.array(x_index).ravel()
    if y_index is not None:
        y_index = np.array(y_index).ravel()

    # считываем данные
    with open(filename, 'r') as f:
        index = f.readline().strip().split(';')[2:]
    data = np.atleast_2d(np.genfromtxt(filename, delimiter=';', dtype=float, encoding='utf-8', skip_header=1))[:, 2:]

    if x_dim is None:
        x_dim = data.shape[1]

    # отделяем индексы
    # index = data[0, :].ravel()

    # data = np.delete(data, 0, axis=0)  # удаляем заголовки
    # data = data.astype(np.float32)

    # определяем позиции нужных индексов
    x_index_number = []
    for i in range(x_index.shape[0]):
        ind = np.where(np.array(index) == np.array(x_index[i]))
        x_index_number.append(ind[0])
    x_index_number = np.array(x_index_number).ravel()
    x_data = data[:, x_index_number].astype("float")

    if y_index is None:
        y_data = data[:, x_dim:]
    else:
        y_index_number = []
        for i in range(y_index.shape[0]):
            ind = np.where(np.array(index) == np.array(y_index[i]))
            y_index_number.append(ind[0])
        # приводим массив индексов к нцжному формату
        y_index_number = np.array(y_index_number).ravel()
        # из данных выбираем нужные столбцы
        y_data = data[:, y_index_number].astype("float")
    return x_data, y_data


def profile(func):
    """
    Для профилирования функции написать перед ней @file_utils.profile
    """

    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result

    return wrapper


# Метод для получения пути к библиотеке
def get_approx_path():
    call_path = os.getcwd()
    separator = None
    for i in range(len(call_path)):
        if call_path[i] == '/' or call_path[i] == '\\':
            if call_path[i + 1] == '/' or call_path[i + 1] == '\\':
                separator = call_path[i] + call_path[i]
            else:
                separator = call_path[i]
            break
    if separator is None:
        raise Exception('Empty separator')
    call_split = call_path.split(sep=separator)
    approx_path = ''
    for item in call_split:
        approx_path += item + separator
        if item == 'approximation-py' or item == 'approximaiton-py':
            break
    return approx_path
