from matplotlib import pyplot as plt

from spectral_reconstruction import (
    reconstruct_spectrum_at_index,
    compare_experimental_vs_reconstructed_at_index,
)


def plot_reconstructed_spectrum(
    T,
    U_prime,
    x_full,
    pack,
    wavelengths,
    idx,
):
    """
    Plotta lo spettro ricostruito a una specifica temperatura.

    Parameters
    ----------
    T : np.ndarray, shape (n_T,)
        Temperature in Kelvin.
    U_prime : np.ndarray, shape (n_wavelengths, n_components)
        Matrice U' usata per la ricostruzione.
    x_full : np.ndarray
        Parametri finali del fit.
    pack : ParamPack
        Struttura con mapping nomi-parametri.
    wavelengths : np.ndarray, shape (n_wavelengths,)
        Griglia delle lunghezze d'onda.
    idx : int
        Indice della temperatura selezionata.
    """
    result = reconstruct_spectrum_at_index(T, U_prime, x_full, pack, idx)

    spectrum = result["spectrum"]
    T_kelvin = result["T_kelvin"]
    T_celsius = T_kelvin - 273.15
    M_vec = result["M"]

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, spectrum, label=f"Reconstructed spectrum at {T_celsius:.1f} °C")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("CD signal")
    plt.title(
        f"Reconstructed spectrum | "
        f"M1={M_vec[0]:.3f}, M2={M_vec[1]:.3f}, M3={M_vec[2]:.3f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spectrum_comparison(
    spectral_matrix,
    spectra_pred,
    T,
    wavelengths,
    idx,
):
    """
    Plotta il confronto tra spettro sperimentale e ricostruito
    alla temperatura selezionata.

    Parameters
    ----------
    spectral_matrix : np.ndarray, shape (n_wavelengths, n_T)
        Matrice spettrale sperimentale.
    spectra_pred : np.ndarray, shape (n_wavelengths, n_T)
        Matrice spettrale ricostruita.
    T : np.ndarray, shape (n_T,)
        Temperature in Kelvin.
    wavelengths : np.ndarray, shape (n_wavelengths,)
        Griglia delle lunghezze d'onda.
    idx : int
        Indice della temperatura selezionata.
    """
    result = compare_experimental_vs_reconstructed_at_index(
        spectral_matrix=spectral_matrix,
        spectra_pred=spectra_pred,
        T=T,
        idx=idx
    )

    T_kelvin = result["T_kelvin"]
    T_celsius = T_kelvin - 273.15
    spectrum_exp = result["spectrum_exp"]
    spectrum_pred = result["spectrum_pred"]
    residual = result["residual"]
    rmse = result["rmse"]
    mae = result["mae"]
    max_abs_error = result["max_abs_error"]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    axes[0].plot(wavelengths, spectrum_exp, "o-", label="Experimental spectrum")
    axes[0].plot(wavelengths, spectrum_pred, "-", label="Reconstructed spectrum")
    axes[0].set_ylabel("CD signal")
    axes[0].set_title(
        f"T = {T_celsius:.2f} °C ({T_kelvin:.2f} K) | "
        f"RMSE = {rmse:.4g} | MAE = {mae:.4g} | MaxAbs = {max_abs_error:.4g}"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(wavelengths, residual, "o-")
    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Exp - Pred")
    axes[1].set_title("Residual spectrum")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()