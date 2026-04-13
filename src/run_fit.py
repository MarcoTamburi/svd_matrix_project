# src/run_fit.py
from datetime import datetime

from fit3 import run_fit3
from fit4 import run_fit4


def run(config_path: str, n_components: int):
    run_metadata = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": config_path,
        "n_components": n_components,
    }

    if n_components == 3:
        return run_fit3(config_path, run_metadata)

    elif n_components == 4:
        return run_fit4(config_path, run_metadata)

    else:
        raise ValueError(f"Numero di componenti non supportato: {n_components}")