from pathlib import Path
import pandas as pd


def extract_temperature_from_column_name(col_name: str) -> int:
    """
    Estrae la temperatura dal nome colonna/file.
    Esempio atteso: qualcosa tipo 'sample_T20' -> 20
    """
    try:
        return int(str(col_name).split("_T")[-1])
    except Exception as e:
        raise ValueError(
            f"Impossibile estrarre la temperatura dal nome '{col_name}'. "
            f"Atteso formato con suffisso tipo '_T20'."
        ) from e


def read_single_cd_file(file_path: Path) -> pd.DataFrame:
    """
    Legge un singolo file raw CD e restituisce un DataFrame con:
    - colonna 'Wavelength'
    - colonna col nome del file (senza estensione)

    Assume che i dati inizino dopo la riga contenente 'XYDATA'
    e che le due colonne utili siano:
    col 0 = wavelength
    col 1 = segnale
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    try:
        start_idx = next(i for i, line in enumerate(lines) if "XYDATA" in line) + 1
    except StopIteration as e:
        raise ValueError(
            f"Nel file '{file_path.name}' non è stata trovata la riga 'XYDATA'."
        ) from e

    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=start_idx,
        header=None,
        usecols=[0, 1],
        names=["Wavelength", file_path.stem],
    )

    return df


def build_temperature_matrix_from_folder(raw_data_folder: str | Path) -> pd.DataFrame:
    """
    Legge tutti i file .txt in una cartella, li unisce sulla colonna Wavelength,
    ordina le colonne per temperatura e aggiunge una prima riga contenente
    le temperature.

    Output finale nel formato:
    riga 0: ['Wavelength', T1, T2, ...]
    righe successive: wavelength + segnali
    """
    raw_data_folder = Path(raw_data_folder)

    if not raw_data_folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {raw_data_folder}")

    file_list = sorted(raw_data_folder.glob("*.txt"))

    if not file_list:
        raise FileNotFoundError(
            f"Nessun file .txt trovato nella cartella: {raw_data_folder}"
        )

    dfs = [read_single_cd_file(file_path) for file_path in file_list]

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Wavelength", how="inner")

    signal_cols = [col for col in merged_df.columns if col != "Wavelength"]

    temp_dict = {
        col: extract_temperature_from_column_name(col)
        for col in signal_cols
    }

    sorted_cols = sorted(signal_cols, key=lambda col: temp_dict[col])
    final_columns = ["Wavelength"] + sorted_cols
    merged_df_sorted = merged_df[final_columns]

    temperature_row = ["Wavelength"] + [temp_dict[col] for col in sorted_cols]
    temp_df = pd.DataFrame([temperature_row], columns=merged_df_sorted.columns)

    merged_df_final = pd.concat([temp_df, merged_df_sorted], ignore_index=True)
    merged_df_final.reset_index(drop=True, inplace=True)

    return merged_df_final


def save_temperature_matrix(df_matrix: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Salva la matrice con T su file CSV-like.
    Mantiene lo stesso formato usato finora nel progetto.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_matrix.to_csv(output_path, index=False, header=False)

    return output_path


def build_clean_svd_input(matrix_path: str | Path) -> pd.DataFrame:
    """
    A partire da matrice_con_t.dat costruisce il DataFrame pulito usato per la SVD:
    - rimuove la prima colonna (wavelength)
    - rimuove la prima riga (temperature)
    - resetta l'indice

    Restituisce il DataFrame pulito, senza salvarlo.
    """
    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(f"File matrice non trovato: {matrix_path}")

    dft_cleaned = pd.read_csv(matrix_path, header=None, sep=",")

    df_clean = dft_cleaned.drop(dft_cleaned.columns[0], axis=1)
    df_clean = df_clean.drop([0])
    df_clean = df_clean.replace(",", ".", regex=True)
    df_clean = df_clean.reset_index(drop=True)

    return df_clean


def save_clean_svd_input(df_clean: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Salva il DataFrame pulito per SVD nel formato TSV senza header.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(output_path, sep="\t", index=False, header=False)

    return output_path


def prepare_cd_inputs(
    raw_data_folder: str | Path,
    matrix_output_path: str | Path,
    clean_output_path: str | Path | None = None,
) -> dict:
    """
    Workflow completo:
    1. costruisce matrice con T dai file raw
    2. salva matrice_con_t.dat
    3. costruisce input pulito per SVD
    4. opzionalmente salva dati_puliti.csv

    Returns
    -------
    dict con:
    - matrix_df
    - clean_df
    - matrix_output_path
    - clean_output_path
    """
    matrix_df = build_temperature_matrix_from_folder(raw_data_folder)
    matrix_output_path = save_temperature_matrix(matrix_df, matrix_output_path)

    clean_df = build_clean_svd_input(matrix_output_path)

    saved_clean_output_path = None
    if clean_output_path is not None:
        saved_clean_output_path = save_clean_svd_input(clean_df, clean_output_path)

    return {
        "matrix_df": matrix_df,
        "clean_df": clean_df,
        "matrix_output_path": matrix_output_path,
        "clean_output_path": saved_clean_output_path,
    }