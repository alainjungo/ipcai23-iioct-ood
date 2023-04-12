import os
import json
import yaml


def get_config(config_file_or_str: str) -> dict:
    if os.path.isfile(config_file_or_str):
        return read_config(config_file_or_str)
    else:
        return json.loads(config_file_or_str)


def read_config(config_file: str) -> dict:
    with open(config_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def save_config(config_params: dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        yaml.dump(config_params, f)


def update_config(params: dict, update_with: dict) -> None:
    params.update(update_with)


def add_config_entries(params: dict, add_from: dict, selection=None) -> None:
    for k, v in add_from.items():
        if k not in params:
            if selection is None or k in selection:
                params[k] = v


def to_str(params: dict) -> str:
    return yaml.dump(params, sort_keys=False)