import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from core.utils import load_params

logger = logging.getLogger(__name__)

def load_patient_file(file_path: Path, hospital: str) -> pd.DataFrame:
    """Carga un archivo .psv de un paciente y añade columnas auxiliares.

    Args:
        file_path (Path): Ruta hacia el archivo .psv
        hospital (str): Hospital de procedencia del paciente

    Returns:
        pd.DataFrame: DataFrame con la información del paciente y variables
        auxiliares
    """
    df_patient = pd.read_csv(file_path, sep="|")

    patient_id = f"{hospital}_{file_path.stem}"

    # Variable auxiliar que identifica al paciente con un id
    df_patient["PatientID"] = patient_id
    # Variable auxiliar que identifica al hospital de procedencia
    df_patient["Hospital"] = hospital

    # TimeStep empieza en 1
    df_patient["TimeStep"] = np.arange(1, len(df_patient) + 1)

    return df_patient


def load_hospital_folder(folder_path: Path, hospital: str) -> pd.DataFrame:
    """Carga todos los archivos .psv de una carpeta correspondiente a un hospital.

    Args:
        folder_path (Path): Ruta hacia la carpeta con los archivos .psv
        hospital (str): Hospital de procedencia de los archivos

    Returns:
        pd.DataFrame: DataFrame con la información de los archivos y variables
        auxiliares

    Raises:
        FileNotFoundError: Si la carpeta no contiene ningún archivo .psv
    """
    files = sorted(folder_path.glob("*.psv"))

    # Si no se encuentran archivos lanza un error
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos .psv en {folder_path}"
        )

    patient_dfs = []

    for i, file_path in enumerate(files, start=1):
        df_patient = load_patient_file(file_path, hospital)
        patient_dfs.append(df_patient)

        # Contador de progreso
        if i % 1000 == 0:
            logger.info("[%s] Cargados %d/%d pacientes...", hospital, i, len(files))

    df_hospital = pd.concat(patient_dfs, ignore_index=True)

    logger.info("[%s] Dataset cargado con éxito. Shape: %s", hospital, df_hospital.shape)
    logger.info("[%s] Número de pacientes únicos: %d", hospital, df_hospital["PatientID"].nunique())

    return df_hospital


def build_combined_dataset(input_a: Path, input_b: Path) -> pd.DataFrame:
    """Integra los conjuntos A y B en un único DataFrame.

    Args:
        input_a (Path): Ruta hacia la carpeta con los archivos procedentes del
          hospital A
        input_b (Path): Ruta hacia la carpeta con los archivos procedentes del
          hospital B

    Returns:
        pd.DataFrame: DataFrame con la información de ambos conjuntos y
        variables auxiliares
    """
    # Cargamos datasets de cada uno de los hospitales
    df_a = load_hospital_folder(input_a, hospital="A")
    df_b = load_hospital_folder(input_b, hospital="B")

    # Concatemos ambos
    df = pd.concat([df_a, df_b], ignore_index=True)

    # Reordenamos columnas auxiliares al principio
    aux_cols = ["PatientID", "Hospital", "TimeStep"]
    other_cols = [col for col in df.columns if col not in aux_cols]
    df = df[aux_cols + other_cols]

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Realiza comprobaciones básicas del dataset integrado.

    Args:
        df (pd.DataFrame): DataFrame con datos combinados del hospital A y el
          hospital B

    Raises:
        ValueError: Si faltan columnas obligatorias en el DataFrame
    """
    # Variables auxiliares
    required_cols = ["PatientID", "Hospital", "TimeStep", "SepsisLabel"]

    # Recopilamos las columnas faltantes si las hay
    missing_cols = [col for col in required_cols if col not in df.columns]

    # Si falta alguna de estas columnas lanza un error
    if missing_cols:
        raise ValueError(f"Faltan columnas obligatorias: {missing_cols}")

    # Se imprimen algunos mensajes de validación
    logger.info("=" * 50)
    logger.info("VALIDACIÓN DEL DATASET INTEGRADO")
    logger.info("=" * 50)
    logger.info("Dimensiones de la matriz (Shape): %s", df.shape)
    logger.info("Pacientes totales en el estudio: %d", df["PatientID"].nunique())
    logger.info("Registros temporales totales (filas): %d", len(df))

    dist_hospital = df.groupby("Hospital")["PatientID"].nunique().to_string()
    logger.info("Distribución de pacientes por Centro:\n%s", dist_hospital)
    
    pos_percentage = df["SepsisLabel"].mean() * 100
    logger.info("Porcentaje de registros positivos (SepsisLabel): %.4f%%", pos_percentage)
    logger.info("=" * 50)


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el dataset integrado en formato Parquet.

    Args:
        df (pd.DataFrame): DataFrame con datos combinados del hospital A y el
          hospital B
        output_path (Path): Ruta completa (incluyendo el nombre del archivo
          .parquet) donde se guardará el dataset
    """
    # Si las carpetas en la ruta no existen, las crea
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convierte el DataFrame a formato Parquet y lo guarda
    df.to_parquet(output_path, index=False)

    logger.info("Dataset integrado guardado con éxito en: %s", output_path)



def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Construye el dataset integrado de PhysioNet Sepsis 2019 a partir de los conjuntos A y B."
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

    data_cfg = params.get("data", {})

    input_a = Path(data_cfg.get("raw_a_dir", "data/raw/training_setA"))
    input_b = Path(data_cfg.get("raw_b_dir", "data/raw/training_setB"))
    output = Path(
        data_cfg.get(
            "combined_path",
            "data/interim/physionet_sepsis_combined.parquet",
        )
    )

    logger.info("Iniciando la construcción del dataset integrado...")
    logger.info("Directorio Origen A: %s", input_a)
    logger.info("Directorio Origen B: %s", input_b)
    logger.info("Archivo de Destino:  %s", output)

    df = build_combined_dataset(input_a, input_b)

    validate_dataset(df)

    save_dataset(df, output)


if __name__ == "__main__":
    main()