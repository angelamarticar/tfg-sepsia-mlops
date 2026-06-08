from pathlib import Path
import argparse
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from core.utils import load_params

logger = logging.getLogger(__name__)

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
        logger.warning("Columnas temporales no encontradas en el dataset y omitidas: %s", missing_temporal_cols)

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

    logger.info("=" * 50)
    logger.info("VALIDACIÓN FINAL DEL PREPROCESAMIENTO")
    logger.info("=" * 50)
    logger.info("Train shape: %s", df_train.shape)
    logger.info("Test shape: %s", df_test.shape)
    logger.info("Pacientes train: %d", df_train['PatientID'].nunique())
    logger.info("Pacientes test: %d", df_test['PatientID'].nunique())
    logger.info("Nulos en predictores train: %d", train_nulls)
    logger.info("Nulos en predictores test: %d", test_nulls)
    logger.info("%% Pacientes con Sepsis (Train): %.2f%%", pct_sepsis_train)
    logger.info("%% Pacientes con Sepsis (Test):  %.2f%%", pct_sepsis_test)
    logger.info("=" * 50)

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

    logger.info("=" * 50)
    logger.info("ARTEFACTOS DE PREPROCESAMIENTO GUARDADOS")
    logger.info("=" * 50)
    for path in saved_paths:
        logger.info("Creado archivo: %s", path)
    logger.info("=" * 50)


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
    logger.info("Cargando dataset unificado desde: %s", input_path)
    df = load_dataset(input_path)

    logger.info("Realizando particionado estratificado por paciente (Test size: %s)...", test_size)
    df_train, df_test = split_by_patient(
        df=df,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info("Eliminando variables con baja representatividad clínica inicial...")
    df_train, df_test, dropped_cols = drop_excluded_columns(df_train, df_test)

    logger.info("Corrigiendo inconsistencias de escala en columna 'FiO2'...")
    df_train, df_test = fix_fio2(df_train, df_test)

    logger.info("Aplicando clipping a valores fisiológicamente extremos/atípicos...")
    df_train, df_test, applied_clipping = apply_clipping(df_train, df_test)

    logger.info("Evaluando y creando indicadores de ausencia (Umbral: %s%%)...", missing_threshold)
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
        logger.warning("Columnas definidas en NO_IMPUTE_COLS ausentes en el espacio de features: %s", unknown_no_impute)

    logger.info("Imputación Fase 1: Aplicando Forward Fill intrapaciente (Series Temporales)...")
    df_train, df_test = temporal_forward_fill(
        df_train=df_train,
        df_test=df_test,
        cols_to_impute=cols_to_impute,
    )

    logger.info("Imputación Fase 2: Rellenando nulos restantes con la mediana calculada de Train...")
    df_train, df_test, medians_train = median_imputation(
        df_train=df_train,
        df_test=df_test,
        cols_to_impute=cols_to_impute,
    )

    logger.info("Ingeniería de Características: Generando medias móviles (%dh) y diferencias (1h)...", rolling_window)
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

    logger.info("Escribiendo y exportando resultados persistentes...")
    save_outputs(
        df_train=df_train,
        df_test=df_test,
        output_dir=output_dir,
        missing_summary=missing_summary,
        medians_train=medians_train,
        metadata=metadata,
    )

    logger.info("Flujo de preprocesamiento completado exitosamente de principio a fin.")



def main() -> None:
    """Punto de entrada principal del script de preprocesamiento."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Aplica el preprocesamiento al dataset integrado y genera los conjuntos de entrenamiento y test."
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="Ruta al archivo params.yaml.",
    )
    args = parser.parse_args()

    params = load_params(args.params)

    log_level = params.get("logging", {}).get("level", "INFO").upper()
    if log_level == "NONE":
        logging.disable(logging.CRITICAL)
    else:
        logging.getLogger().setLevel(log_level)

    logger.info("Nivel de logging configurado: %s", log_level)
    logger.debug("params.yaml cargado desde: %s", args.params)

    data_cfg = params.get("data", {})
    preprocessing_cfg = params.get("preprocessing", {})

    input_path = Path(
        data_cfg.get("combined_path", "data/interim/physionet_sepsis_combined.parquet")
    )

    output_dir = Path(
        data_cfg.get("processed_dir", "data/processed")
    )

    test_size = float(
        preprocessing_cfg.get("test_size", 0.2)
    )

    random_state = int(
        params.get("random_state", 42)
    )

    missing_threshold = float(
        preprocessing_cfg.get("missing_threshold", 70.0)
    )

    rolling_window = int(
        preprocessing_cfg.get("rolling_window", 6)
    )

    run_preprocessing(
        input_path=input_path,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
        missing_threshold=missing_threshold,
        rolling_window=rolling_window,
    )


if __name__ == "__main__":
    main()