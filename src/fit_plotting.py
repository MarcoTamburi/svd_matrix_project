import pandas as pd
from matplotlib import pyplot as plt

from model_fit3 import predict_vprime_from_params


def save_final_fit_plots(out_dir, T, V_prime, x_full, pack):
    _, _, f_pred = predict_vprime_from_params(T, x_full, pack)

    labels = ["V1_prime", "V2_prime", "V3_prime"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axs[i].plot(T - 273.15, V_prime[i], "o", label="Experimental data")
        axs[i].plot(T - 273.15, f_pred[i], "-", label="Global fit")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend()

    axs[2].set_xlabel("Temperature (°C)")
    plt.suptitle("Final global fit")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "final_global_fit.png", dpi=300)
    plt.close()

    return f_pred


def save_final_fit_data(out_dir, T, V_prime, f_pred):
    df = pd.DataFrame({
        "T_kelvin": T,
        "T_celsius": T - 273.15,
        "V1_exp": V_prime[0],
        "V1_fit": f_pred[0],
        "V1_resid": V_prime[0] - f_pred[0],
        "V2_exp": V_prime[1],
        "V2_fit": f_pred[1],
        "V2_resid": V_prime[1] - f_pred[1],
        "V3_exp": V_prime[2],
        "V3_fit": f_pred[2],
        "V3_resid": V_prime[2] - f_pred[2],
    })

    df.to_csv(out_dir / "final_fit_curves.csv", index=False)