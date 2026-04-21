from pathlib import Path
import pandas as pd
import numpy as np


def load_user_matrix(matrix_path: str | Path) -> pd.DataFrame:
    """
    Legge una matrice spettrale già pronta fornita dall'utente.

    Formato atteso:
    - file .csv
    - prima riga = header
    - prima colonna = Wavelength
    - colonne successive = temperature numeriche
    """
    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(f"File matrice non trovato: {matrix_path}")

    df = pd.read_csv(matrix_path)

    return df


def validate_user_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida il formato della matrice utente.

    Requisiti:
    - almeno 2 colonne
    - prima colonna chiamata 'Wavelength'
    - intestazioni successive convertibili a float
    - tutti i valori numerici
    - nessun NaN

    Restituisce una copia pulita del DataFrame.
    """
    if df.shape[1] < 2:
        raise ValueError(
            "La matrice deve contenere almeno 2 colonne: "
            "'Wavelength' + almeno una temperatura."
        )

    first_col = str(df.columns[0]).strip()
    if first_col.lower() != "wavelength":
        raise ValueError(
            f"La prima colonna deve chiamarsi 'Wavelength', trovata: '{df.columns[0]}'"
        )

    df_clean = df.copy()

    # Controllo nomi colonne temperatura
    temperature_cols = df_clean.columns[1:]
    try:
        temperatures = [float(col) for col in temperature_cols]
    except Exception as e:
        raise ValueError(
            "Le intestazioni delle colonne dopo 'Wavelength' devono essere temperature numeriche, "
            "per esempio: 20, 25, 30 oppure 20.0, 25.0, 30.0"
        ) from e

    # Controllo colonna wavelength numerica
    try:
        df_clean.iloc[:, 0] = pd.to_numeric(df_clean.iloc[:, 0], errors="raise")
    except Exception as e:
        raise ValueError("La colonna 'Wavelength' deve contenere solo valori numerici.") from e

    # Controllo segnali numerici
    for col in temperature_cols:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="raise")
        except Exception as e:
            raise ValueError(
                f"La colonna '{col}' contiene valori non numerici."
            ) from e

    # Controllo NaN
    if df_clean.isnull().any().any():
        nan_rows, nan_cols = np.where(df_clean.isnull())
        raise ValueError(
            f"La matrice contiene valori mancanti (NaN). "
            f"Primo NaN trovato in riga {nan_rows[0]}, colonna {nan_cols[0]}."
        )

    # Ordinamento opzionale per wavelength crescente
    df_clean = df_clean.sort_values(by="Wavelength").reset_index(drop=True)

    # Riordino colonne per temperatura crescente
    sorted_temp_cols = [str(t) for t in sorted(temperatures)]

    # Mappa robusta nome originale -> float -> nome standard
    temp_map = {col: float(col) for col in temperature_cols}
    rename_map = {col: str(temp_map[col]) for col in temperature_cols}

    df_clean = df_clean.rename(columns=rename_map)
    df_clean = df_clean[["Wavelength"] + sorted_temp_cols]

    return df_clean


def save_validated_matrix(df_matrix: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Salva la matrice validata nel formato CSV con header.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_matrix.to_csv(output_path, index=False)

    return output_path


def build_clean_svd_input_from_df(df_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce il DataFrame pulito usato per la SVD:
    - rimuove la colonna 'Wavelength'
    - mantiene solo i segnali
    - nessun header nel file finale di output SVD

    Restituisce il DataFrame pulito senza salvarlo.
    """
    if "Wavelength" not in df_matrix.columns:
        raise ValueError("Colonna 'Wavelength' non trovata nella matrice.")

    df_clean = df_matrix.drop(columns=["Wavelength"]).copy()
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean


def build_clean_svd_input(matrix_path: str | Path) -> pd.DataFrame:
    """
    Workflow compatto:
    1. legge la matrice utente
    2. la valida
    3. costruisce il DataFrame pulito per la SVD

    Restituisce il DataFrame pulito, senza salvarlo.
    """
    df_matrix = load_user_matrix(matrix_path)
    df_matrix = validate_user_matrix(df_matrix)
    df_clean = build_clean_svd_input_from_df(df_matrix)

    return df_clean


def save_clean_svd_input(df_clean: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Salva il DataFrame pulito per la SVD nel formato TSV senza header.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(output_path, sep="\t", index=False, header=False)

    return output_path


def prepare_matrix_inputs(
    matrix_path: str | Path,
    validated_matrix_output_path: str | Path | None = None,
    clean_output_path: str | Path | None = None,
) -> dict:
    """
    Workflow completo per il nuovo progetto matrix-first:

    1. legge la matrice utente
    2. valida la matrice
    3. opzionalmente salva una copia validata/standardizzata
    4. costruisce il DataFrame pulito per la SVD
    5. opzionalmente salva il file pulito per la SVD

    Returns
    -------
    dict con:
    - matrix_df
    - clean_df
    - validated_matrix_output_path
    - clean_output_path
    """
    matrix_df = load_user_matrix(matrix_path)
    matrix_df = validate_user_matrix(matrix_df)

    saved_matrix_path = None
    if validated_matrix_output_path is not None:
        saved_matrix_path = save_validated_matrix(matrix_df, validated_matrix_output_path)

    clean_df = build_clean_svd_input_from_df(matrix_df)

    saved_clean_output_path = None
    if clean_output_path is not None:
        saved_clean_output_path = save_clean_svd_input(clean_df, clean_output_path)

    return {
        "matrix_df": matrix_df,
        "clean_df": clean_df,
        "validated_matrix_output_path": saved_matrix_path,
        "clean_output_path": saved_clean_output_path,
    }