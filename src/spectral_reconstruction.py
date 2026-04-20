import numpy as np

from model_fit3 import (
    build_C_matrix as build_C_matrix_fit3,
    predict_vprime_from_params as predict_vprime_from_params_fit3,
)
from model_fit4 import (
    build_C_matrix as build_C_matrix_fit4,
    predict_vprime_from_params as predict_vprime_from_params_fit4,
)


def get_n_components_from_pack(pack):
    c_names = [name for name in pack.names if name.startswith("C")]
    if not c_names:
        raise ValueError("Impossibile determinare n_components: nessun coefficiente C trovato.")

    max_index = 0
    for name in c_names:
        if len(name) != 3 or not name[1:].isdigit():
            continue
        i = int(name[1])
        j = int(name[2])
        max_index = max(max_index, i, j)

    if max_index not in (3, 4):
        raise ValueError(f"Numero di componenti non supportato nella reconstruction: {max_index}")

    return max_index


def get_model_functions(pack):
    n_components = get_n_components_from_pack(pack)

    if n_components == 3:
        return build_C_matrix_fit3, predict_vprime_from_params_fit3
    elif n_components == 4:
        return build_C_matrix_fit4, predict_vprime_from_params_fit4
    else:
        raise ValueError(f"Numero di componenti non supportato: {n_components}")


def reconstruct_state_spectra(U_prime, x_full, pack):
    """
    Ricostruisce gli spettri puri di tutti gli stati termodinamici.

    Parameters
    ----------
    U_prime : np.ndarray, shape (n_wavelengths, n_components)
    x_full : np.ndarray
    pack : ParamPack

    Returns
    -------
    dict
        {
            "state_spectra": {
                "state_1": array (...,),
                ...
            },
            "C": array (n_components, n_components)
        }
    """
    build_C_matrix, _ = get_model_functions(pack)
    C = build_C_matrix(x_full, pack)

    n_components = C.shape[1]
    state_spectra = {}

    for j in range(n_components):
        state_spectra[f"state_{j+1}"] = U_prime @ C[:, j]

    return {
        "state_spectra": state_spectra,
        "C": C,
    }


def compute_populations(T, x_full, pack):
    """
    Calcola le popolazioni M(T) di tutti gli stati.
    """
    _, predict_vprime_from_params = get_model_functions(pack)
    _, M, _ = predict_vprime_from_params(T, x_full, pack)
    return M


def reconstruct_all_spectra(T, U_prime, x_full, pack):
    """
    Ricostruisce tutti gli spettri su tutta la griglia di temperatura.
    """
    _, predict_vprime_from_params = get_model_functions(pack)
    _, M, V_pred = predict_vprime_from_params(T, x_full, pack)
    spectra_pred = U_prime @ V_pred

    state_result = reconstruct_state_spectra(U_prime, x_full, pack)

    return {
        "spectra_pred": spectra_pred,
        "V_pred": V_pred,
        "M": M,
        "state_spectra": state_result["state_spectra"],
        "C": state_result["C"],
    }


def reconstruct_spectrum_at_index(T, U_prime, x_full, pack, idx):
    """
    Ricostruisce lo spettro a una specifica temperatura identificata da indice.
    """
    result = reconstruct_all_spectra(T, U_prime, x_full, pack)

    spectra_pred = result["spectra_pred"]
    V_pred = result["V_pred"]
    M = result["M"]

    return {
        "T_kelvin": float(T[idx]),
        "M": M[:, idx],
        "V": V_pred[:, idx],
        "spectrum": spectra_pred[:, idx],
    }


def compare_experimental_vs_reconstructed_at_index(
    spectral_matrix,
    spectra_pred,
    T,
    idx
):
    """
    Estrae spettro sperimentale, spettro ricostruito, residuo e metriche
    per una specifica temperatura identificata da idx.
    """
    if idx < 0 or idx >= spectral_matrix.shape[1]:
        raise IndexError(
            f"idx fuori range: {idx}. "
            f"Valori ammessi: 0 - {spectral_matrix.shape[1] - 1}"
        )

    if spectral_matrix.shape != spectra_pred.shape:
        raise ValueError(
            f"Shape incompatibili tra spectral_matrix e spectra_pred: "
            f"{spectral_matrix.shape} vs {spectra_pred.shape}"
        )

    if spectral_matrix.shape[1] != len(T):
        raise ValueError(
            f"Numero di colonne di spectral_matrix ({spectral_matrix.shape[1]}) "
            f"diverso dalla lunghezza di T ({len(T)})"
        )

    spectrum_exp = spectral_matrix[:, idx]
    spectrum_pred = spectra_pred[:, idx]
    residual = spectrum_exp - spectrum_pred

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs_error = float(np.max(np.abs(residual)))

    return {
        "idx": int(idx),
        "T_kelvin": float(T[idx]),
        "spectrum_exp": spectrum_exp,
        "spectrum_pred": spectrum_pred,
        "residual": residual,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
    }


def compute_reconstruction_metrics_over_T(
    spectral_matrix,
    spectra_pred,
    T
):
    """
    Calcola le metriche di ricostruzione per tutte le temperature.
    """
    if spectral_matrix.shape != spectra_pred.shape:
        raise ValueError(
            f"Shape incompatibili tra spectral_matrix e spectra_pred: "
            f"{spectral_matrix.shape} vs {spectra_pred.shape}"
        )

    if spectral_matrix.shape[1] != len(T):
        raise ValueError(
            f"Numero di colonne di spectral_matrix ({spectral_matrix.shape[1]}) "
            f"diverso dalla lunghezza di T ({len(T)})"
        )

    n_temperatures = spectral_matrix.shape[1]

    rmse_values = []
    mae_values = []
    max_abs_error_values = []

    for idx in range(n_temperatures):
        comparison = compare_experimental_vs_reconstructed_at_index(
            spectral_matrix=spectral_matrix,
            spectra_pred=spectra_pred,
            T=T,
            idx=idx
        )

        rmse_values.append(comparison["rmse"])
        mae_values.append(comparison["mae"])
        max_abs_error_values.append(comparison["max_abs_error"])

    return {
        "T_kelvin": np.array(T, dtype=float),
        "T_celsius": np.array(T, dtype=float) - 273.15,
        "rmse": np.array(rmse_values, dtype=float),
        "mae": np.array(mae_values, dtype=float),
        "max_abs_error": np.array(max_abs_error_values, dtype=float),
    }

import pandas as pd


def build_transition_summary_df(x_full, pack):
    """
    Costruisce una tabella riassuntiva delle transizioni:
    - Tm assolute (K e °C)
    - Delta H

    NON modifica nulla del modello o dei parametri salvati.
    Serve solo per visualizzazione.
    """
    def get(name):
        return float(x_full[pack.name_to_i[name]])

    names = set(pack.names)
    rows = []

    # Caso con deltaTm (nuovo modello)
    if "dTm12" in names:
        Tm1 = get("Tm1")
        dTm12 = get("dTm12")

        Tm2 = Tm1 + dTm12

        rows.append({
            "Transition": "1",
            "Tm (K)": Tm1,
            "Tm (°C)": Tm1 - 273.15,
            "ΔH (cal/mol)": get("dH1")
        })

        rows.append({
            "Transition": "2",
            "Tm (K)": Tm2,
            "Tm (°C)": Tm2 - 273.15,
            "ΔH (cal/mol)": get("dH2")
        })

        # Fit4
        if "dTm23" in names:
            dTm23 = get("dTm23")
            Tm3 = Tm2 + dTm23

            rows.append({
                "Transition": "3",
                "Tm (K)": Tm3,
                "Tm (°C)": Tm3 - 273.15,
                "ΔH (cal/mol)": get("dH3")
            })

    # Fallback (vecchio modello)
    else:
        if "Tm1" in names:
            rows.append({
                "Transition": "1",
                "Tm (K)": get("Tm1"),
                "Tm (°C)": get("Tm1") - 273.15,
                "ΔH (cal/mol)": get("dH1")
            })

        if "Tm2" in names:
            rows.append({
                "Transition": "2",
                "Tm (K)": get("Tm2"),
                "Tm (°C)": get("Tm2") - 273.15,
                "ΔH (cal/mol)": get("dH2")
            })

        if "Tm3" in names:
            rows.append({
                "Transition": "3",
                "Tm (K)": get("Tm3"),
                "Tm (°C)": get("Tm3") - 273.15,
                "ΔH (cal/mol)": get("dH3")
            })

    df = pd.DataFrame(rows)

    # opzionale: arrotondamento più leggibile
    df["Tm (K)"] = df["Tm (K)"].round(2)
    df["Tm (°C)"] = df["Tm (°C)"].round(2)
    df["ΔH (cal/mol)"] = df["ΔH (cal/mol)"].round(2)

    return df