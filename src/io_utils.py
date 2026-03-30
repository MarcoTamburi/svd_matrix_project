# src/io_utils.py
import pandas as pd
import numpy as np


def load_fit3_inputs(
    spectra_matrix_path: str,
    V_prime_path: str,
    U_prime_path: str
):
    # matrice spettrale:
    # - prima riga = intestazioni con le T
    # - prima colonna = wavelength
    # - resto = segnali spettrali
    df_spectra = pd.read_csv(spectra_matrix_path,sep=",", header=0)

    # T ricavata dai nomi colonna, saltando la prima colonna
    columns = df_spectra.columns.tolist()
    T = np.array([float(c) for c in columns[1:]], dtype=float)

    # matrice spettrale numerica senza la prima colonna
    spectral_matrix = df_spectra.iloc[:, 1:].to_numpy(dtype=float)

    # lunghezze d'onda
    wavelengths = df_spectra.iloc[:, 0].to_numpy(dtype=float)

    # V_prime
    V_prime = pd.read_csv(
        V_prime_path,
        sep="\t",
        header=None
    ).to_numpy(dtype=float)

    # U_prime
    U_prime = pd.read_csv(
        U_prime_path,
        sep="\t",
        header=None
    ).to_numpy(dtype=float)

    return T, V_prime, U_prime, spectral_matrix, wavelengths