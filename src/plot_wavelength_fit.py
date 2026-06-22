"""
plot_wavelength_fit.py

Legge user_spectral_matrix.csv, estrae il segnale a lunghezze d'onda
specifiche in funzione della temperatura, e fitta ciascuna curva con
un modello sigmoide a 2 stati:

    f(T) = yF + (yU - yF) / (1 + exp(dH * (1 - T/Tm) / (R * T)))

Produce un plot con 3 pannelli (uno per lambda) e segna le Tm con
una linea verticale tratteggiata.

Uso:
    python plot_wavelength_fit.py

Parametri iniziali e lambda modificabili in fondo al file.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 1.987  # cal/(mol*K)


# ──────────────────────────────────────────────────────────────────────────────
# Modello
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid_2state(T_kelvin, yF, yU, Tm, dH):
    """
    Modello a 2 stati in funzione della temperatura (Kelvin).

        f(T) = yF + (yU - yF) / (1 + exp(dH * (1 - T/Tm) / (R * T)))
    """
    exponent = dH * (1.0 - T_kelvin / Tm) / (R * T_kelvin)
    return yF + (yU - yF) / (1.0 + np.exp(exponent))


def residuals(params, T_kelvin, signal):
    yF, yU, Tm, dH = params
    return sigmoid_2state(T_kelvin, yF, yU, Tm, dH) - signal


# ──────────────────────────────────────────────────────────────────────────────
# Caricamento dati
# ──────────────────────────────────────────────────────────────────────────────

def load_matrix(matrix_path: Path) -> pd.DataFrame:
    df = pd.read_csv(matrix_path)
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    return df


def extract_signal_at_wavelength(df: pd.DataFrame, target_wl: float) -> tuple:
    """
    Estrae il segnale alla wavelength più vicina a target_wl.
    Restituisce (T_array_celsius, signal_array, wavelength_effettiva).
    """
    wl_col = df.columns[0]
    wl_array = df[wl_col].to_numpy(dtype=float)

    idx = np.argmin(np.abs(wl_array - target_wl))
    actual_wl = wl_array[idx]

    T_celsius = np.array([float(c) for c in df.columns[1:]], dtype=float)
    signal = df.iloc[idx, 1:].to_numpy(dtype=float)

    return T_celsius, signal, actual_wl


# ──────────────────────────────────────────────────────────────────────────────
# Fit
# ──────────────────────────────────────────────────────────────────────────────

def fit_sigmoid(T_celsius, signal, p0: dict) -> dict:
    """
    Fitta il modello sigmoide sui dati.

    p0 deve contenere le chiavi: yF, yU, Tm (in K), dH
    """
    T_kelvin = T_celsius + 273.15

    x0 = [p0["yF"], p0["yU"], p0["Tm"], p0["dH"]]

    # Bounds generosi: lasciano libertà al fit ma evitano divergenze
    lower = [-np.inf, -np.inf, 273.15, 100.0]
    upper = [np.inf,  np.inf,  473.15, 200000.0]

    try:
        result = least_squares(
            residuals,
            x0,
            args=(T_kelvin, signal),
            bounds=(lower, upper),
            method="trf",
        )
        yF_fit, yU_fit, Tm_fit, dH_fit = result.x
        success = result.success
    except Exception as e:
        print(f"  Fit fallito: {e}")
        yF_fit, yU_fit, Tm_fit, dH_fit = x0
        success = False

    T_fine = np.linspace(T_celsius.min(), T_celsius.max(), 500)
    T_fine_K = T_fine + 273.15
    signal_fit = sigmoid_2state(T_fine_K, yF_fit, yU_fit, Tm_fit, dH_fit)

    return {
        "yF": yF_fit,
        "yU": yU_fit,
        "Tm_K": Tm_fit,
        "Tm_C": Tm_fit - 273.15,
        "dH": dH_fit,
        "success": success,
        "T_fine": T_fine,
        "signal_fit": signal_fit,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_fits(data_list, out_path: Path):
    """
    data_list: lista di dict con chiavi:
        wl, T_celsius, signal, fit_result
    """
    n = len(data_list)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)

    if n == 1:
        axs = [axs]

    colors = ["#2c7bb6", "#d7191c", "#1a9641"]

    for ax, item, color in zip(axs, data_list, colors):
        wl = item["wl"]
        T = item["T_celsius"]
        sig = item["signal"]
        fit = item["fit_result"]

        # Dati sperimentali
        ax.plot(T, sig, "o", color=color, markersize=4,
                alpha=0.7, label=f"Data λ={wl:.1f} nm")

        # Curva fittata
        ax.plot(fit["T_fine"], fit["signal_fit"], "-", color=color,
                linewidth=2, label="2-state fit")

        # Linea verticale Tm
        Tm_C = fit["Tm_C"]
        ax.axvline(x=Tm_C, color="black", linestyle="--", linewidth=1.5,
                   label=f"Tm = {Tm_C:.1f} °C")

        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("CD signal", fontsize=12)
        ax.set_title(f"λ = {wl:.1f} nm", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("2-state sigmoid fit at selected wavelengths", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main — modifica qui i parametri iniziali e le lambda
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Path al file matrice
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MATRIX_PATH = PROJECT_ROOT / "data" / "user_spectral_matrix.csv"
    OUT_PATH = PROJECT_ROOT / "results" / "wavelength_fit.png"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Lambda da plottare (nm) ──────────────────────────────────────────────
    WAVELENGTHS = [245.0, 260.0, 290.0]

    # ── Parametri iniziali del fit ───────────────────────────────────────────
    # Tm in Kelvin: se pensi che la transizione sia ~62°C → Tm = 335 K
    # Modifica questi valori in base a quello che vedi nei dati raw
    P0 = {
        "yF":  30.0,     # segnale allo stato folded (T bassa)
        "yU":   5.0,     # segnale allo stato unfolded (T alta)
        "Tm":  335.0,    # temperatura di melting iniziale (Kelvin)
        "dH":  20000.0,  # entalpia iniziale (cal/mol)
    }

    # ── Esecuzione ───────────────────────────────────────────────────────────
    df = load_matrix(MATRIX_PATH)
    print(f"Matrice caricata: {df.shape[0]} wavelength, {df.shape[1]-1} temperature")

    data_list = []
    for target_wl in WAVELENGTHS:
        T_celsius, signal, actual_wl = extract_signal_at_wavelength(df, target_wl)
        print(f"\nλ = {actual_wl:.1f} nm")

        fit_result = fit_sigmoid(T_celsius, signal, P0)

        print(f"  Tm = {fit_result['Tm_C']:.2f} °C")
        print(f"  dH = {fit_result['dH']:.0f} cal/mol")
        print(f"  yF = {fit_result['yF']:.4f},  yU = {fit_result['yU']:.4f}")
        print(f"  Fit success: {fit_result['success']}")

        data_list.append({
            "wl": actual_wl,
            "T_celsius": T_celsius,
            "signal": signal,
            "fit_result": fit_result,
        })

    plot_fits(data_list, OUT_PATH)