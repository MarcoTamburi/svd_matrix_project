import numpy as np

from model_fit3 import calc_M_2p, build_C_matrix, predict_vprime_from_params


def reconstruct_state_spectra(U_prime, x_full, pack):
    """
    Ricostruisce gli spettri puri dei tre stati:
    folded, intermediate, unfolded.

    Parameters
    ----------
    U_prime : np.ndarray, shape (n_wavelengths, 3)
    x_full : np.ndarray
    pack : ParamPack

    Returns
    -------
    dict
        {
            "s_f": array (n_wavelengths,),
            "s_i": array (n_wavelengths,),
            "s_u": array (n_wavelengths,),
            "C": array (3, 3)
        }
    """
    C = build_C_matrix(x_full, pack)

    s_f = U_prime @ C[:, 0]
    s_i = U_prime @ C[:, 1]
    s_u = U_prime @ C[:, 2]

    return {
        "s_f": s_f,
        "s_i": s_i,
        "s_u": s_u,
        "C": C,
    }


def compute_populations(T, x_full, pack):
    """
    Calcola le popolazioni M(T) dei tre stati.

    Parameters
    ----------
    T : np.ndarray, shape (n_T,)
        Temperature in Kelvin
    x_full : np.ndarray
    pack : ParamPack

    Returns
    -------
    np.ndarray, shape (3, n_T)
        M(T) = [M1, M2, M3]
    """
    Tm1 = float(x_full[pack.name_to_i["Tm1"]])
    Tm2 = float(x_full[pack.name_to_i["Tm2"]])
    dH1 = float(x_full[pack.name_to_i["dH1"]])
    dH2 = float(x_full[pack.name_to_i["dH2"]])

    M = calc_M_2p(T, Tm1, Tm2, dH1, dH2)
    return M


def reconstruct_all_spectra(T, U_prime, x_full, pack):
    """
    Ricostruisce tutti gli spettri su tutta la griglia di temperatura.

    Parameters
    ----------
    T : np.ndarray, shape (n_T,)
        Temperature in Kelvin
    U_prime : np.ndarray, shape (n_wavelengths, 3)
    x_full : np.ndarray
    pack : ParamPack

    Returns
    -------
    dict
        {
            "spectra_pred": array (n_wavelengths, n_T),
            "V_pred": array (3, n_T),
            "M": array (3, n_T),
            "state_spectra": {
                "s_f": ...,
                "s_i": ...,
                "s_u": ...
            }
        }
    """
    _, M, V_pred = predict_vprime_from_params(T, x_full, pack)
    spectra_pred = U_prime @ V_pred

    state_spectra = reconstruct_state_spectra(U_prime, x_full, pack)

    return {
        "spectra_pred": spectra_pred,
        "V_pred": V_pred,
        "M": M,
        "state_spectra": {
            "s_f": state_spectra["s_f"],
            "s_i": state_spectra["s_i"],
            "s_u": state_spectra["s_u"],
        }
    }


def reconstruct_spectrum_at_index(T, U_prime, x_full, pack, idx):
    """
    Ricostruisce lo spettro a una specifica temperatura identificata da indice.

    Parameters
    ----------
    T : np.ndarray, shape (n_T,)
        Temperature in Kelvin
    U_prime : np.ndarray, shape (n_wavelengths, 3)
    x_full : np.ndarray
    pack : ParamPack
    idx : int
        Indice della temperatura selezionata

    Returns
    -------
    dict
        {
            "T_kelvin": float,
            "M": array (3,),
            "V": array (3,),
            "spectrum": array (n_wavelengths,)
        }
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

    Parameters
    ----------
    spectral_matrix : np.ndarray, shape (n_wavelengths, n_T)
        Matrice spettrale sperimentale.
    spectra_pred : np.ndarray, shape (n_wavelengths, n_T)
        Matrice spettrale ricostruita dal modello.
    T : np.ndarray, shape (n_T,)
        Temperature in Kelvin.
    idx : int
        Indice della temperatura selezionata.

    Returns
    -------
    dict
        {
            "idx": int,
            "T_kelvin": float,
            "spectrum_exp": array (n_wavelengths,),
            "spectrum_pred": array (n_wavelengths,),
            "residual": array (n_wavelengths,),
            "rmse": float,
            "mae": float,
            "max_abs_error": float
        }
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

    Parameters
    ----------
    spectral_matrix : np.ndarray, shape (n_wavelengths, n_T)
        Matrice spettrale sperimentale.
    spectra_pred : np.ndarray, shape (n_wavelengths, n_T)
        Matrice spettrale ricostruita dal modello.
    T : np.ndarray, shape (n_T,)
        Temperature in Kelvin.

    Returns
    -------
    dict
        {
            "T_kelvin": array (n_T,),
            "T_celsius": array (n_T,),
            "rmse": array (n_T,),
            "mae": array (n_T,),
            "max_abs_error": array (n_T,)
        }
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