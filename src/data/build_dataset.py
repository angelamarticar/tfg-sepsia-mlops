import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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
            print(f"[{hospital}] Cargados {i}/{len(files)} pacientes")

    df_hospital = pd.concat(patient_dfs, ignore_index=True)

    print(f"[{hospital}] Dataset cargado: {df_hospital.shape}")
    print(f"[{hospital}] Pacientes: {df_hospital['PatientID'].nunique()}")

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
    print("\nValidación del dataset integrado")
    print("--------------------------------")
    print(f"Shape: {df.shape}")
    print(f"Pacientes totales: {df['PatientID'].nunique()}")
    print(f"Registros totales: {len(df)}")
    print("Distribución por hospital:")
    print(df.groupby("Hospital")["PatientID"].nunique())
    print("Porcentaje de registros positivos:")
    print(df["SepsisLabel"].mean() * 100)


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

    print(f"\nDataset guardado en: {output_path}")


def parse_args() -> argparse.Namespace:
    """Configura y lee los argumentos de la línea de comandos.

    Returns:
        argparse.Namespace: Objeto con los argumentos validados
    """
    parser = argparse.ArgumentParser(
        description="Construye el dataset integrado de PhysioNet Sepsis 2019."
    )

    parser.add_argument(
        "--input-a",
        type=Path,
        required=True,
        help="Ruta a la carpeta training_setA con archivos .psv",
    )

    parser.add_argument(
        "--input-b",
        type=Path,
        required=True,
        help="Ruta a la carpeta training_setB con archivos .psv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/physionet_sepsis_combined.parquet"),
        help="Ruta de salida del dataset integrado en formato parquet",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Construyendo dataset integrado...")
    print(f"Input A: {args.input_a}")
    print(f"Input B: {args.input_b}")
    print(f"Output: {args.output}")

    df = build_combined_dataset(args.input_a, args.input_b)

    validate_dataset(df)

    save_dataset(df, args.output)


if __name__ == "__main__":
    main()