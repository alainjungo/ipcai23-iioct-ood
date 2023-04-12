import time


def get_unique_identifier() -> str:
    return time.strftime('%y%m%d-%H%M%S')
