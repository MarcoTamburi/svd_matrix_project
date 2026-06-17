from pathlib import Path

from fit3 import load_config
from io_utils import load_fit_inputs
from params_utils import read_params_file


def find_latest_run(results_root):
    """
    Trova la run completata più recente dentro:
    results/fit2_run/
    results/fit3_run/
    results/fit4_run/
    """
    results_root = Path(results_root)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root non trovata: {results_root}")

    run_dirs = []

    for fit_dir in results_root.glob("fit*_run"):
        if not fit_dir.is_dir():
            continue

        for run_dir in fit_dir.iterdir():
            if (
                run_dir.is_dir()
                and (run_dir / "config_used.json").exists()
                and (
                    (run_dir / "params_final.xlsx").exists()
                    or (run_dir / "params_final.xls").exists()
                    or (run_dir / "params_final.csv").exists()
                )
            ):
                run_dirs.append(run_dir)

    if not run_dirs:
        raise FileNotFoundError(
            f"Nessuna run completata trovata dentro: {results_root}"
        )

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return latest_run



def load_latest_completed_run(results_root):
    """
    Trova e carica automaticamente l'ultima run completata,
    indipendentemente dal numero di componenti.
    """
    latest_run_dir = find_latest_run(results_root)
    return load_completed_run(latest_run_dir)

def _find_params_final_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "params_final.xlsx",
        run_dir / "params_final.xls",
        run_dir / "params_final.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"params_final non trovato in: {run_dir}")


def load_completed_run(run_dir):
    """
    Carica una run già completata senza rieseguire il fit.
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory non trovata: {run_dir}")

    config_used_path = run_dir / "config_used.json"
    reconstruction_dir = run_dir / "reconstruction"

    if not config_used_path.exists():
        raise FileNotFoundError(f"config_used.json non trovato in: {run_dir}")

    params_final_path = _find_params_final_file(run_dir)

    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_used_path))

    # I path nel config usato sono relativi alla cartella configs/
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "configs"

    pack = read_params_file(str(params_final_path))

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

