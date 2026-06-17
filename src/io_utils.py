# src/io_utils.py
# src/io_utils.py
import pandas as pd
import numpy as np


def load_fit_inputs(
    spectra_matrix_path: str,
    V_prime_path: str,
    U_prime_path: str
):
    df_spectra = pd.read_csv(spectra_matrix_path, sep=",", header=0)

    # NUOVO: riordina per wavelength crescente, altrimenti le righe di
    # spectral_matrix non corrispondono a quelle di U_prime (che viene
    # dalla SVD di dati_puliti.csv, ordinato crescente da validate_user_matrix)
    df_spectra = df_spectra.sort_values(by=df_spectra.columns[0]).reset_index(drop=True)


    T = np.array([float(c) for c in df_spectra.columns[1:]], dtype=float)

    wavelengths = df_spectra.iloc[:, 0].to_numpy(dtype=float)
    spectral_matrix = df_spectra.iloc[:, 1:].to_numpy(dtype=float)

    V_prime = pd.read_csv(
        V_prime_path,
        sep="\t",
        header=None
    ).to_numpy(dtype=float)

    U_prime = pd.read_csv(
        U_prime_path,
        sep="\t",
        header=None
    ).to_numpy(dtype=float)

    return T, V_prime, U_prime, spectral_matrix, wavelengths