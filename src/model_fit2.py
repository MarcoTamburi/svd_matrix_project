import numpy as np

R = 1.987  # cal/(mol*K)


def calc_M_2s(T, Tm, dH):
    """
    Popolazioni per modello a 2 stati:

        S1 <-> S2

    Parameters
    ----------
    T : np.ndarray
        Temperature in Kelvin
    Tm : float
        Melting temperature in Kelvin
    dH : float
        Enthalpy in cal/mol

    Returns
    -------
    np.ndarray
        M(T), shape (2, n_T)
        righe: [M1, M2]
    """
    A = np.exp(-dH / R * (1 / Tm - 1 / T))

    denom = 1 + A

    M1 = 1 / denom
    M2 = A / denom

    return np.stack([M1, M2], axis=0)


def build_C_matrix(x_full, pack):
    """
    Costruisce la matrice C 2x2 dai parametri.
    """
    def get(name):
        return float(x_full[pack.name_to_i[name]])

    C = np.array([
        [get("C11"), get("C12")],
        [get("C21"), get("C22")],
    ], dtype=float)

    return C


def predict_vprime_from_params(T, x_full, pack):
    def get(name):
        return float(x_full[pack.name_to_i[name]])

    Tm = get("Tm1")
    dH = get("dH1")

    C = build_C_matrix(x_full, pack)
    M = calc_M_2s(T, Tm, dH)
    V_pred = C @ M

    return C, M, V_pred


def residuals_fit2(x_full, T, V_prime, pack):
    """
    Residui flattenati tra V'_exp e V'_pred.
    """
    _, _, f_pred = predict_vprime_from_params(T, x_full, pack)
    resid = (V_prime - f_pred).flatten()

    if np.any(np.isnan(resid)) or np.any(np.isinf(resid)):
        return np.full(V_prime.size, 1e12, dtype=float)

    return resid