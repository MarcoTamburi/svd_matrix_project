from pathlib import Path

from fit3 import load_config
from io_utils import load_fit_inputs
from params_utils import read_params_csv


def find_latest_run(results_dir, n_components):
    """
    Trova la run più recente dentro results/fit{n_components}_run.
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory non trovata: {results_dir}")

    prefix = f"fit{n_components}_"
    run_dirs = [
        p for p in results_dir.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]

    if not run_dirs:
        raise FileNotFoundError(
            f"Nessuna cartella di run trovata in: {results_dir}"
        )

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return latest_run


def load_completed_run(run_dir):
    """
    Carica una run già completata senza rieseguire il fit.
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory non trovata: {run_dir}")

    config_used_path = run_dir / "config_used.json"
    params_final_path = run_dir / "params_final.csv"
    reconstruction_dir = run_dir / "reconstruction"

    if not config_used_path.exists():
        raise FileNotFoundError(f"config_used.json non trovato in: {run_dir}")

    if not params_final_path.exists():
        raise FileNotFoundError(f"params_final.csv non trovato in: {run_dir}")

    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_used_path))

    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "configs"

    pack = read_params_csv(str(params_final_path))

    spectra_matrix_path = (config_dir / cfg["data"]["spectra_matrix_path"]).resolve()
    v_prime_path = (config_dir / cfg["data"]["V_prime_path"]).resolve()
    u_prime_path = (config_dir / cfg["data"]["U_prime_path"]).resolve()

    T, V_prime, U_prime, spectral_matrix, wavelengths = load_fit_inputs(
        str(spectra_matrix_path),
        str(v_prime_path),
        str(u_prime_path),
    )

    T = T + 273.15
    x_final = pack.x0_full.copy()

    return {
        "run_dir": run_dir,
        "reconstruction_dir": reconstruction_dir,
        "cfg": cfg,
        "pack": pack,
        "x_final": x_final,
        "T": T,
        "V_prime": V_prime,
        "U_prime": U_prime,
        "spectral_matrix": spectral_matrix,
        "wavelengths": wavelengths,
    }

def load_latest_completed_run(results_dir, n_components):
    """
    Trova e carica automaticamente la run più recente per il numero di componenti scelto.
    """
    latest_run_dir = find_latest_run(results_dir, n_components)
    return load_completed_run(latest_run_dir)