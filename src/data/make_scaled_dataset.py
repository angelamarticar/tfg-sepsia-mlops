from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from core.utils import load_params

TARGET_COL = "SepsisLabel"
AUX_COLS = ["PatientID", "Hospital", "TimeStep"]

def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datasets preprocesados en formato parquet.

    Args:
        train_path (Path): Ruta al dataset de entrenamiento.
        test_path (Path): Ruta al dataset de prueba.
    Returns:
        tuple: (df_train, df_test) con los datos cargados.
    Raises:
        FileNotFoundError: Si alguno de los archivos no existe.
    """
    if not train_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrenamiento: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"No existe el archivo de prueba: {test_path}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    return df_train, df_test

_NON_FEATURE_COLS = AUX_COLS + [TARGET_COL]

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Obtiene las columnas predictoras excluyendo auxiliares y variable objetivo.

    Args:
        df (pd.DataFrame): DataFrame del que extraer las columnas predictoras.
    Returns:
        list[str]: Lista de nombres de columnas predictoras.
    """
    return [col for col in df.columns if col not in _NON_FEATURE_COLS]

def identify_binary_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Identifica columnas binarias que no deben escalarse.

    Una columna se considera binaria si sus únicos valores no nulos son 0 y/o 1.
    Incluye indicadores de ausencia y variables como Gender.

    Args:
        df (pd.DataFrame): DataFrame sobre el que identificar columnas binarias.
        feature_cols (list[str]): Columnas predictoras a inspeccionar.
    Returns:
        list[str]: Lista de columnas binarias encontradas.
    """
    binary_cols = []
    for col in feature_cols:
        if set(df[col].dropna().unique()).issubset({0, 1}):
            binary_cols.append(col)
    return binary_cols

def build_scaled_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, dict]:
    """Escala las variables continuas con RobustScaler y conserva las binarias sin escalar.

    El scaler se ajusta exclusivamente sobre df_train para evitar data leakage
    y se aplica a df_test. Las columnas auxiliares y el target se conservan
    en el DataFrame de salida para trazabilidad.

    Args:
        df_train (pd.DataFrame): Conjunto de entrenamiento preprocesado.
        df_test (pd.DataFrame): Conjunto de prueba preprocesado.
    Returns:
        tuple: (df_train_scaled, df_test_scaled, preprocessor, metadata) donde
               preprocessor es el ColumnTransformer ajustado sobre train.
    """
    feature_cols = get_feature_columns(df_train)
    binary_cols = identify_binary_columns(df_train, feature_cols)
    numeric_cols_to_scale = [col for col in feature_cols if col not in binary_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("robust_scaler", RobustScaler(), numeric_cols_to_scale),
            ("binary_features", "passthrough", binary_cols),
        ],
        remainder="drop",
    )

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    y_train = df_train[TARGET_COL].reset_index(drop=True)
    y_test = df_test[TARGET_COL].reset_index(drop=True)

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    scaled_feature_names = numeric_cols_to_scale + binary_cols

    X_train_scaled_df = pd.DataFrame(
    data=np.array(X_train_scaled, dtype=float),
    columns=scaled_feature_names,
    index=df_train.index,
    )
    X_test_scaled_df = pd.DataFrame(
    data=np.array(X_test_scaled, dtype=float),
    columns=scaled_feature_names,
    index=df_test.index,
    )

    available_aux_cols = [col for col in AUX_COLS if col in df_train.columns]

    df_train_scaled = pd.concat(
        [
            df_train[available_aux_cols].reset_index(drop=True),
            X_train_scaled_df.reset_index(drop=True),
            y_train,
        ],
        axis=1,
    )
    df_test_scaled = pd.concat(
        [
            df_test[available_aux_cols].reset_index(drop=True),
            X_test_scaled_df.reset_index(drop=True),
            y_test,
        ],
        axis=1,
    )

    metadata = {
        "target_col": TARGET_COL,
        "aux_cols": available_aux_cols,
        "feature_cols": feature_cols,
        "numeric_cols_scaled": numeric_cols_to_scale,
        "binary_cols_not_scaled": binary_cols,
        "scaler": "RobustScaler",
        "scaler_fit_on": "train",
        "n_features": len(feature_cols),
        "n_numeric_scaled": len(numeric_cols_to_scale),
        "n_binary_not_scaled": len(binary_cols),
    }

    return df_train_scaled, df_test_scaled, preprocessor, metadata


def validate_scaled_outputs(
    df_train_scaled: pd.DataFrame,
    df_test_scaled: pd.DataFrame,
) -> None:
    """Comprueba dimensiones y ausencia de nulos en los datasets escalados.

    Args:
        df_train_scaled (pd.DataFrame): Conjunto de entrenamiento escalado.
        df_test_scaled (pd.DataFrame): Conjunto de prueba escalado.
    Raises:
        ValueError: Si quedan nulos en alguna columna predictora tras el escalado.
    """
    feature_cols = get_feature_columns(df_train_scaled)
    train_nulls = df_train_scaled[feature_cols].isna().sum().sum()
    test_nulls = df_test_scaled[feature_cols].isna().sum().sum()

    print("\nValidación del escalado")
    print("-----------------------")
    print(f"Train scaled shape: {df_train_scaled.shape}")
    print(f"Test scaled shape: {df_test_scaled.shape}")
    print(f"Nulos en predictores train scaled: {train_nulls}")
    print(f"Nulos en predictores test scaled: {test_nulls}")

    if train_nulls > 0 or test_nulls > 0:
        raise ValueError(
            f"Quedan nulos tras el escalado. "
            f"Train: {train_nulls}, Test: {test_nulls}"
        )
    
def save_outputs(
    df_train_scaled: pd.DataFrame,
    df_test_scaled: pd.DataFrame,
    preprocessor: ColumnTransformer,
    metadata: dict,
    output_dir: Path,
) -> None:
    """Guarda los datasets escalados, el transformador ajustado y el metadata.

    Args:
        df_train_scaled (pd.DataFrame): Conjunto de entrenamiento escalado.
        df_test_scaled (pd.DataFrame): Conjunto de prueba escalado.
        preprocessor (ColumnTransformer): Transformador ajustado sobre train.
        metadata (dict): Diccionario con los parámetros y decisiones del escalado.
        output_dir (Path): Directorio donde se guardarán todos los archivos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_scaled_path = output_dir / "train_preprocessed_scaled.parquet"
    test_scaled_path = output_dir / "test_preprocessed_scaled.parquet"
    preprocessor_path = output_dir / "robust_scaler_preprocessor.joblib"
    metadata_path = output_dir / "scaling_metadata.json"

    df_train_scaled.to_parquet(train_scaled_path, index=False)
    df_test_scaled.to_parquet(test_scaled_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    saved_paths = [train_scaled_path, test_scaled_path, preprocessor_path, metadata_path]
    print("\nArchivos guardados")
    print("------------------")
    for path in saved_paths:
        print(path)

def run_scaling(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
) -> None:
    """Ejecuta el flujo completo de escalado.

    Args:
        train_path (Path): Ruta al dataset de entrenamiento preprocesado.
        test_path (Path): Ruta al dataset de prueba preprocesado.
        output_dir (Path): Directorio donde se guardarán los resultados.
    """
    print("Cargando datasets preprocesados...")
    df_train, df_test = load_data(train_path, test_path)

    print("Aplicando escalado robusto a variables continuas...")
    df_train_scaled, df_test_scaled, preprocessor, metadata = build_scaled_datasets(
        df_train=df_train,
        df_test=df_test,
    )

    validate_scaled_outputs(df_train_scaled, df_test_scaled)

    print("Guardando datasets escalados...")
    save_outputs(
        df_train_scaled=df_train_scaled,
        df_test_scaled=df_test_scaled,
        preprocessor=preprocessor,
        metadata=metadata,
        output_dir=output_dir,
    )

    print("\nEscalado completado correctamente.")

def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Genera versiones escaladas de los conjuntos preprocesados para modelos sensibles a la escala."
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="Ruta al archivo params.yaml.",
    )
    return parser.parse_args()

def main() -> None:
    """Punto de entrada principal del script de escalado."""
    args = parse_args()
    params = load_params(args.params)

    data_cfg = params.get("data", {})

    train_path = Path(
        data_cfg.get("train_path", "data/processed/train_preprocessed.parquet")
    )

    test_path = Path(
        data_cfg.get("test_path", "data/processed/test_preprocessed.parquet")
    )

    output_dir = Path(
        data_cfg.get("processed_dir", "data/processed")
    )

    print("Ejecutando escalado...")
    print(f"Params: {args.params}")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print(f"Output dir: {output_dir}")

    run_scaling(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()