import numpy as np

R = 1.987  # gas constant in cal/(mol*K)


def calc_M_4s(T, Tm1, Tm2, Tm3, dH1, dH2, dH3):
    """
    Calcola le popolazioni dei 4 stati per un modello sequenziale:

        S1 <-> S2 <-> S3 <-> S4

    Parameters
    ----------
    T : np.ndarray
        Temperature in Kelvin, shape (n_T,)
    Tm1, Tm2, Tm3 : float
        Temperature di transizione in Kelvin
    dH1, dH2, dH3 : float
        Entalpie di transizione in cal/mol

    Returns
    -------
    np.ndarray
        Popolazioni M(T), shape (4, n_T)
        con righe [M1, M2, M3, M4]
    """
    A = np.exp(-dH1 / R * (1 / Tm1 - 1 / T))
    B = np.exp(-dH2 / R * (1 / Tm2 - 1 / T))
    C = np.exp(-dH3 / R * (1 / Tm3 - 1 / T))

    denom = 1 + A + A * B + A * B * C

    M1 = 1 / denom
    M2 = A / denom
    M3 = A * B / denom
    M4 = A * B * C / denom

    return np.stack([M1, M2, M3, M4], axis=0)


def build_C_matrix(x_full, pack):
    """
    Costruisce la matrice C 4x4 dai parametri del fit.

    Returns
    -------
    np.ndarray
        Matrice C di shape (4, 4)
    """
    def get(name):
        return float(x_full[pack.name_to_i[name]])

    C = np.array([
        [get("C11"), get("C12"), get("C13"), get("C14")],
        [get("C21"), get("C22"), get("C23"), get("C24")],
        [get("C31"), get("C32"), get("C33"), get("C34")],
        [get("C41"), get("C42"), get("C43"), get("C44")],
    ], dtype=float)

    return C



def predict_vprime_from_params(T, x_full, pack):
    def get(name):
        return float(x_full[pack.name_to_i[name]])

    Tm1 = get("Tm1")
    Tm2 = get("Tm2")
    Tm3 = get("Tm3")
    dH1 = get("dH1")
    dH2 = get("dH2")
    dH3 = get("dH3")

    C = build_C_matrix(x_full, pack)
    M = calc_M_4s(T, Tm1, Tm2, Tm3, dH1, dH2, dH3)
    V_pred = C @ M

    return C, M, V_pred


def residuals_fit4(x_full, T, V_prime, pack):
    """
    Restituisce i residui flattenati tra V'_sperimentale e V'_predetto.

    Parameters
    ----------
    x_full : np.ndarray
        Vettore completo dei parametri
    T : np.ndarray
        Temperature in Kelvin, shape (n_T,)
    V_prime : np.ndarray
        Dati sperimentali, shape (4, n_T)
    pack : ParamPack
        Struttura parametri

    Returns
    -------
    np.ndarray
        Residui flattenati, shape (4 * n_T,)
    """
    _, _, f_pred = predict_vprime_from_params(T, x_full, pack)
    resid = (V_prime - f_pred).flatten()

    if np.any(np.isnan(resid)) or np.any(np.isinf(resid)):
        return np.full(V_prime.size, 1e12, dtype=float)

    return resid