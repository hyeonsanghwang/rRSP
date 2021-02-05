import os


def data_path(name):
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, 'D:/respiration/')
    return os.path.join(path, name)


def model_path(name):
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, '../model')
    return os.path.join(path, name)


def result_path(name):
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, '../result')
    return os.path.join(path, name)


def ui_path(name):
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, '../ui')
    return os.path.join(path, name)


def src_path(name):
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, '../src')
    return os.path.join(path, name)