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