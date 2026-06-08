from pathlib import Path
import json
import yaml
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    """Lee un archivo YAML de configuración y lo devuelve como diccionario.

    Args:
        path (str): Ruta al archivo YAML. Por defecto "params.yaml".

    Returns:
        dict: Diccionario con todos los parámetros del archivo. Las secciones
        anidadas se convierten en diccionarios anidados.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(paths):
    """Crea los directorios indicados si no existen.

    Args:
        paths (list[str | Path]): Lista de rutas de directorios a crear.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    """Guarda un objeto Python como archivo JSON.

    Args:
        data (dict): Objeto a serializar en JSON.
        path (str | Path): Ruta donde se guardará el archivo JSON. Si los
        directorios intermedios no existen, se crean automáticamente.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def make_json_serializable(obj):
    """Convierte recursivamente un objeto a tipos serializables por JSON.

    Transforma tipos de NumPy (enteros, flotantes, arrays) y colecciones
    anidadas (dict, list, tuple) a sus equivalentes nativos de Python.

    Args:
        obj: Objeto a convertir. Puede ser dict, list, tuple, np.integer,
        np.floating, np.ndarray o cualquier otro tipo ya serializable.

    Returns:
        Versión del objeto con todos los tipos compatibles con JSON.
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(make_json_serializable(value) for value in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_model_artifacts(
    model,
    model_name: str,
    threshold: float,
    metrics_row: dict,
    metrics_patient: dict,
    output_dirs: dict,
    params: dict | None = None,
) -> None:
    """Guarda el modelo entrenado, el umbral, las métricas y los parámetros asociados.

    Serializa el modelo en un archivo .joblib y escribe un archivo JSON con
    los metadatos (umbral, métricas y parámetros). Los directorios de salida
    se crean automáticamente si no existen.

    Args:
        model: Modelo entrenado compatible con joblib (scikit-learn, XGBoost,
        LightGBM, etc.).
        model_name (str): Nombre legible del modelo. Se normaliza para usarlo
        como nombre de archivo.
        threshold (float): Umbral de clasificación óptimo seleccionado durante
        el entrenamiento.
        metrics_row (dict): Métricas calculadas a nivel de fila o instancia.
        metrics_patient (dict): Métricas calculadas a nivel de paciente.
        output_dirs (dict): Diccionario con las rutas de salida. Debe contener
        la clave "models_dir".
        params (dict | None): Hiperparámetros del modelo. Si es None, se
        guarda un diccionario vacío.
    """
    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )

    models_dir = Path(output_dirs["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{safe_name}.joblib"
    metadata_path = models_dir / f"{safe_name}_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
        "metrics_row": metrics_row,
        "metrics_patient": metrics_patient,
        "params": params if params is not None else {},
    }

    metadata = make_json_serializable(metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info("Modelo guardado en: %s", model_path)
    logger.info("Metadatos guardados en: %s", metadata_path)