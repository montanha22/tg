import os

import yaml

_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def get_root_path(path: str):
    return os.path.join(_ROOT, path)


def get_data_path(path: str):
    return os.path.join(_ROOT, 'src', 'data', path)


def change_mlflow_yaml():
    file_location = os.path.join(_ROOT, 'mlruns', '0', 'meta.yaml')

    with open(file_location) as f:
        mlflow_yaml = yaml.safe_load(f)

    aux = os.path.dirname(file_location)
    if aux.startswith('/'):
        aux = aux[1:]
    mlflow_yaml['artifact_location'] = 'file:///{}'.format(aux)

    with open(file_location, "w") as f:
        yaml.dump(mlflow_yaml, f)
