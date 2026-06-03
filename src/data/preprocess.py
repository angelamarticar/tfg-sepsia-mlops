from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COL = "SepsisLabel"
AUX_COLS = ["PatientID", "Hospital", "TimeStep"]
_NON_FEATURE_COLS = AUX_COLS + [TARGET_COL]

DROP_COLS_BASE = [
    "EtCO2",
    "HCO3",
    "SaO2",
    "AST",
    "Alkalinephos",
    "Bilirubin_direct",
    "TroponinI",
    "Hgb",
    "PTT",
    "Fibrinogen",
    "Unit1",
    "Unit2",
]

NO_IMPUTE_COLS = [
    "Age",
    "Gender",
    "ICULOS",
]

TEMPORAL_COLS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "Resp",
]

# Rangos de clipping fisiológico. None indica que no se aplica límite en ese extremo.
CLIPPING_RANGES = {
    "Calcium":    (None, 20),
    "Creatinine": (None, 20),
    "BaseExcess": (-32, 45),
    "Glucose":    (None, 800),
    "Phosphate":  (None, 18),
}


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Carga el dataset integrado generado por build_dataset.py.

    Args:
        input_path (Path): Ruta hacia el dataset en formato Parquet.
    Returns:
        pd.DataFrame: DataFrame con los registros cargados del parquet.
    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
        ValueError: Si faltan columnas obligatorias en el dataset.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_parquet(input_path)

    required_cols = AUX_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas obligatorias: {missing_cols}")

    return df


def split_by_patient(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train/test a nivel de paciente, estratificando por sepsis.

    Args:
        df (pd.DataFrame): DataFrame con todos los registros a nivel de timestep.
        test_size (float): Proporción de pacientes destinada al conjunto de prueba.
        random_state (int): Semilla aleatoria para reproducibilidad.
    Returns:
        tuple: (df_train, df_test).
    """
    df_patient = (
        df.groupby("PatientID")
        .agg(
            TieneSepsis=(TARGET_COL, "max"),
            Hospital=("Hospital", "first"),
            NumRegistros=(TARGET_COL, "size"),
        )
        .reset_index()
    )

    train_patients, test_patients = train_test_split(
        df_patient["PatientID"],
        test_size=test_size,
        random_state=random_state,
        stratify=df_patient["TieneSepsis"],
    )

    df_train = df[df["PatientID"].isin(train_patients)].copy()
    df_test = df[df["PatientID"].isin(test_patients)].copy()

    return df_train, df_test


def drop_excluded_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Elimina las variables descartadas en la fase de preprocesamiento.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento.
        df_test (pd.DataFrame): Conjunto de prueba.
    Returns:
        tuple: (df_train, df_test, cols_to_drop) donde cols_to_drop es la lista
               de columnas efectivamente eliminadas.
    """
    cols_to_drop = [col for col in DROP_COLS_BASE if col in df_train.columns]
    df_train = df_train.drop(columns=cols_to_drop)
    df_test = df_test.drop(columns=cols_to_drop)
    return df_train, df_test, cols_to_drop


def fix_fio2(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrige FiO2 expresada como porcentaje y elimina valores fuera de rango fisiológico.

    Algunos registros expresan FiO2 como porcentaje (1-100) en lugar de fracción (0.21-1.0).
    Los valores entre 1 y 100 se dividen entre 100. Los valores fuera del rango
    fisiológico plausible tras la corrección se tratan como ausentes.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento.
        df_test (pd.DataFrame): Conjunto de prueba.
    Returns:
        tuple: (df_train, df_test) con FiO2 corregida.
    """
    for df in [df_train, df_test]:
        if "FiO2" in df.columns:
            mask_percent = (df["FiO2"] > 1) & (df["FiO2"] <= 100)
            df.loc[mask_percent, "FiO2"] = df.loc[mask_percent, "FiO2"] / 100

            mask_invalid = (df["FiO2"] < 0.21) | (df["FiO2"] > 1)
            df.loc[mask_invalid, "FiO2"] = np.nan

    return df_train, df_test


def apply_clipping(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aplica clipping con rangos fisiológicos definidos en CLIPPING_RANGES.

    None en cualquiera de los extremos indica que no se aplica límite en ese lado.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento.
        df_test (pd.DataFrame): Conjunto de prueba.
    Returns:
        tuple: (df_train, df_test, applied_ranges) donde applied_ranges es un
               diccionario con los rangos efectivamente aplicados por columna,
               con la forma {col: {"lower": ..., "upper": ...}}.
    """
    applied_ranges = {}
    for col, (lower, upper) in CLIPPING_RANGES.items():
        for df in [df_train, df_test]:
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        if col in df_train.columns:
            applied_ranges[col] = {"lower": lower, "upper": upper}
    return df_train, df_test, applied_ranges


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Obtiene las columnas predictoras excluyendo auxiliares y variable objetivo.

    Args:
        df (pd.DataFrame): DataFrame del que extraer las columnas predictoras.
    Returns:
        list[str]: Lista de nombres de columnas predictoras.
    """
    return [col for col in df.columns if col not in _NON_FEATURE_COLS]


def create_missing_indicators(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    missing_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Crea indicadores de ausencia para variables con porcentaje de nulos superior al umbral.

    Los umbrales se calculan exclusivamente sobre df_train para evitar data leakage.
    Las mismas columnas se aplican a df_test independientemente de su tasa de nulos.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento.
        df_test (pd.DataFrame): Conjunto de prueba.
        feature_cols (list[str]): Columnas predictoras sobre las que calcular ausencias.
        missing_threshold (float): Porcentaje mínimo de nulos para crear el indicador.
    Returns:
        tuple: (df_train, df_test, missing_summary, indicator_cols) donde
               missing_summary contiene el porcentaje de ausentes por variable en train
               e indicator_cols es la lista de columnas indicadoras creadas.
    """
    missing_train = df_train[feature_cols].isna().mean().sort_values(ascending=False) * 100
    high_missing_cols = missing_train[missing_train > missing_threshold].index.tolist()

    missing_summary = missing_train.reset_index()
    missing_summary.columns = ["Variable", "PorcentajeAusentesTrain"]

    indicator_cols = []
    for col in high_missing_cols:
        indicator_col = f"{col}_missing"
        df_train[indicator_col] = df_train[col].isna().astype(int)
        df_test[indicator_col] = df_test[col].isna().astype(int)
        indicator_cols.append(indicator_col)

    return df_train, df_test, missing_summary, indicator_cols


def temporal_forward_fill(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols_to_impute: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica forward fill dentro de cada paciente para imputar valores ausentes.

    El orden por PatientID y TimeStep se garantiza internamente antes de aplicar
    el relleno, por lo que no es necesario que el DataFrame de entrada esté ordenado.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento.
        df_test (pd.DataFrame): Conjunto de prueba.
        cols_to_impute (list[str]): Columnas sobre las que aplicar el forward fill.
    Returns:
        tuple: (df_train, df_test) con los valores ausentes rellenados hacia adelante
               dentro de cada paciente.
    """
    df_train = df_train.sort_values(["PatientID", "TimeStep"])
    df_test = df_test.sort_values(["PatientID", "TimeStep"])

    df_train[cols_to_impute] = df_train.groupby("PatientID")[cols_to_impute].ffill()
    df_test[cols_to_impute] = df_test.groupby("PatientID")[cols_to_impute].ffill()

    return df_train, df_test


def median_imputation(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols_to_impute: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Imputa los nulos restantes con la mediana global de entrenamiento.

    Las medianas se calculan exclusivamente sobre df_train para evitar data leakage
    y se aplican tanto a df_train como a df_test.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento.
        df_test (pd.DataFrame): Conjunto de prueba.
        cols_to_impute (list[str]): Columnas sobre las que aplicar la imputación.
    Returns:
        tuple: (df_train, df_test, medians_train) donde medians_train es una Serie
               con la mediana de cada columna calculada sobre el conjunto de entrenamiento.
    Raises:
        ValueError: Si alguna columna no tiene mediana calculable (todos los valores son nulos).
    """
    medians_train = df_train[cols_to_impute].median()

    cols_without_median = medians_train[medians_train.isna()].index.tolist()
    if cols_without_median:
        raise ValueError(
            "Hay columnas sin mediana calculable. Revisa estas variables: "
            f"{cols_without_median}"
        )

    df_train[cols_to_impute] = df_train[cols_to_impute].fillna(medians_train)
    df_test[cols_to_impute] = df_test[cols_to_impute].fillna(medians_train)

    return df_train, df_test, medians_train


def add_temporal_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    temporal_cols: list[str],
    rolling_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Genera medias móviles y diferencias de primer orden por paciente.

    Las transformaciones se aplican dentro de cada paciente respetando el orden
    temporal. Las columnas de temporal_cols que no existan en el DataFrame
    se omiten con un aviso.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento.
        df_test (pd.DataFrame): Conjunto de prueba.
        temporal_cols (list[str]): Columnas sobre las que generar features temporales.
        rolling_window (int): Tamaño de la ventana para la media móvil (en horas).
    Returns:
        tuple: (df_train, df_test, created_cols) donde created_cols es la lista
               de nombres de las nuevas columnas generadas.
    """
    df_train = df_train.sort_values(["PatientID", "TimeStep"])
    df_test = df_test.sort_values(["PatientID", "TimeStep"])

    available_temporal_cols = [col for col in temporal_cols if col in df_train.columns]
    missing_temporal_cols = [col for col in temporal_cols if col not in df_train.columns]
    if missing_temporal_cols:
        print(f"[WARN] Columnas temporales no encontradas y omitidas: {missing_temporal_cols}")

    def _add_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
        rolling_col = f"{col}_rolling_mean_{rolling_window}h"
        diff_col = f"{col}_diff_1h"
        df[rolling_col] = (
            df.groupby("PatientID")[col]
            .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
        )
        df[diff_col] = (
            df.groupby("PatientID")[col]
            .diff()
            .fillna(0)
        )
        return df

    created_cols = []
    for col in available_temporal_cols:
        df_train = _add_features(df_train, col)
        df_test = _add_features(df_test, col)
        created_cols.extend([f"{col}_rolling_mean_{rolling_window}h", f"{col}_diff_1h"])

    return df_train, df_test, created_cols


def validate_outputs(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Comprueba que los datasets finales no tienen nulos en columnas predictoras.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento preprocesado.
        df_test (pd.DataFrame): Conjunto de prueba preprocesado.
    Raises:
        ValueError: Si quedan nulos en alguna columna predictora tras el preprocesamiento.
    """
    feature_cols = get_feature_columns(df_train)
    train_nulls = df_train[feature_cols].isna().sum().sum()
    test_nulls = df_test[feature_cols].isna().sum().sum()

    pct_sepsis_train = df_train.groupby("PatientID")[TARGET_COL].max().mean() * 100
    pct_sepsis_test = df_test.groupby("PatientID")[TARGET_COL].max().mean() * 100

    print("\nValidación final")
    print("----------------")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"Pacientes train: {df_train['PatientID'].nunique()}")
    print(f"Pacientes test: {df_test['PatientID'].nunique()}")
    print(f"Nulos en predictores train: {train_nulls}")
    print(f"Nulos en predictores test: {test_nulls}")
    print(f"% pacientes con sepsis train: {pct_sepsis_train:.2f}")
    print(f"% pacientes con sepsis test: {pct_sepsis_test:.2f}")

    if train_nulls > 0 or test_nulls > 0:
        raise ValueError(
            f"Quedan nulos tras el preprocesamiento. "
            f"Train: {train_nulls}, Test: {test_nulls}"
        )


def save_outputs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    missing_summary: pd.DataFrame,
    medians_train: pd.Series,
    metadata: dict,
) -> None:
    """Guarda datasets preprocesados y artefactos auxiliares en el directorio de salida.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento preprocesado.
        df_test (pd.DataFrame): Conjunto de prueba preprocesado.
        output_dir (Path): Directorio donde se guardarán todos los archivos.
        missing_summary (pd.DataFrame): Resumen de porcentajes de ausentes por variable.
        medians_train (pd.Series): Medianas de imputación calculadas sobre train.
        metadata (dict): Diccionario con los parámetros y decisiones del preprocesamiento.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_preprocessed.parquet"
    test_path = output_dir / "test_preprocessed.parquet"
    missing_path = output_dir / "missing_decisions.csv"
    medians_path = output_dir / "imputation_medians_train.csv"
    features_path = output_dir / "feature_columns.json"
    metadata_path = output_dir / "preprocessing_metadata.json"

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    missing_summary.to_csv(missing_path, index=False)
    medians_train.to_csv(medians_path, header=["median"])

    feature_cols = get_feature_columns(df_train)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4, ensure_ascii=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    saved_paths = [train_path, test_path, missing_path, medians_path, features_path, metadata_path]
    print("\nArchivos guardados")
    print("------------------")
    for path in saved_paths:
        print(path)


def run_preprocessing(
    input_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    missing_threshold: float,
    rolling_window: int,
) -> None:
    """Ejecuta el flujo completo de preprocesamiento.

    Args:
        input_path (Path): Ruta al dataset integrado en formato parquet.
        output_dir (Path): Directorio donde se guardarán los resultados.
        test_size (float): Proporción de pacientes destinada al conjunto de prueba.
        random_state (int): Semilla aleatoria para reproducibilidad.
        missing_threshold (float): Porcentaje de nulos a partir del cual se crea
                                   indicador de ausencia.
        rolling_window (int): Tamaño de la ventana para las medias móviles (en horas).
    """
    print("Cargando dataset integrado...")
    df = load_dataset(input_path)

    print("Realizando particionado por paciente...")
    df_train, df_test = split_by_patient(
        df=df,
        test_size=test_size,
        random_state=random_state,
    )

    print("Eliminando variables descartadas...")
    df_train, df_test, dropped_cols = drop_excluded_columns(df_train, df_test)

    print("Corrigiendo FiO2...")
    df_train, df_test = fix_fio2(df_train, df_test)

    print("Aplicando clipping a valores fisiológicamente poco plausibles...")
    df_train, df_test, applied_clipping = apply_clipping(df_train, df_test)

    print("Creando indicadores de ausencia...")
    feature_cols_before_indicators = get_feature_columns(df_train)
    df_train, df_test, missing_summary, indicator_cols = create_missing_indicators(
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols_before_indicators,
        missing_threshold=missing_threshold,
    )

    cols_to_impute = [
        col for col in feature_cols_before_indicators
        if col not in NO_IMPUTE_COLS
    ]
    unknown_no_impute = [col for col in NO_IMPUTE_COLS if col not in feature_cols_before_indicators]
    if unknown_no_impute:
        print(f"[WARN] NO_IMPUTE_COLS no encontradas en features: {unknown_no_impute}")

    print("Aplicando forward fill por paciente...")
    df_train, df_test = temporal_forward_fill(
        df_train=df_train,
        df_test=df_test,
        cols_to_impute=cols_to_impute,
    )

    print("Aplicando imputación con mediana de entrenamiento...")
    df_train, df_test, medians_train = median_imputation(
        df_train=df_train,
        df_test=df_test,
        cols_to_impute=cols_to_impute,
    )

    print("Generando características temporales...")
    df_train, df_test, temporal_created_cols = add_temporal_features(
        df_train=df_train,
        df_test=df_test,
        temporal_cols=TEMPORAL_COLS,
        rolling_window=rolling_window,
    )

    metadata = {
        "input_path": str(input_path),
        "test_size": test_size,
        "random_state": random_state,
        "missing_threshold": missing_threshold,
        "rolling_window": rolling_window,
        "target_col": TARGET_COL,
        "aux_cols": AUX_COLS,
        "dropped_cols": dropped_cols,
        "no_impute_cols": NO_IMPUTE_COLS,
        "cols_to_impute": cols_to_impute,
        "missing_indicator_cols": indicator_cols,
        "temporal_cols": TEMPORAL_COLS,
        "temporal_created_cols": temporal_created_cols,
        "clipping_ranges": applied_clipping,
        "n_patients_total": int(df["PatientID"].nunique()),
        "n_patients_train": int(df_train["PatientID"].nunique()),
        "n_patients_test": int(df_test["PatientID"].nunique()),
        "n_rows_train": int(len(df_train)),
        "n_rows_test": int(len(df_test)),
    }

    validate_outputs(df_train, df_test)

    print("Guardando resultados...")
    save_outputs(
        df_train=df_train,
        df_test=df_test,
        output_dir=output_dir,
        missing_summary=missing_summary,
        medians_train=medians_train,
        metadata=metadata,
    )

    print("\nPreprocesamiento completado correctamente.")


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Objeto con los argumentos parseados.
    """
    def proportion(value: str) -> float:
        v = float(value)
        if not 0 < v < 1:
            raise argparse.ArgumentTypeError(
                f"Debe ser un valor entre 0 y 1 (exclusivo). Recibido: {v}"
            )
        return v

    def percentage(value: str) -> float:
        v = float(value)
        if not 0 <= v <= 100:
            raise argparse.ArgumentTypeError(
                f"Debe ser un porcentaje entre 0 y 100. Recibido: {v}"
            )
        return v

    parser = argparse.ArgumentParser(
        description="Aplica el preprocesamiento base al dataset integrado de PhysioNet Sepsis 2019."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/physionet_sepsis_combined.parquet"),
        help="Ruta al dataset integrado en formato parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directorio donde se guardarán los datasets preprocesados.",
    )
    parser.add_argument(
        "--test-size",
        type=proportion,
        default=0.2,
        help="Proporción de pacientes destinada al conjunto de prueba (entre 0 y 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=percentage,
        default=70.0,
        help="Porcentaje de valores ausentes a partir del cual se crea indicador de ausencia (0-100).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=6,
        help="Tamaño de la ventana temporal para medias móviles.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal del script de preprocesamiento."""
    args = parse_args()
    run_preprocessing(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        missing_threshold=args.missing_threshold,
        rolling_window=args.rolling_window,
    )


if __name__ == "__main__":
    main()