from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Configuración del logger local de este módulo
logger = logging.getLogger(__name__)


class SepsisPredictor:
    """Carga el modelo final y genera predicciones de riesgo de sepsis de forma eficiente.

    """

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        """Inicializa el predictor cargando el modelo y sus metadatos asociados.

        Args:
            model_path (str | Path): Ruta al archivo comprimido del modelo (.joblib).
            metadata_path (str | Path): Ruta al archivo JSON de metadatos.

        Raises:
            ValueError: Si los metadatos no contienen la lista de variables predictoras.
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        logger.info("Inicializando SepsisPredictor...")
        self.model = self._load_model(self.model_path)
        self.metadata = self._load_metadata(self.metadata_path)

        self.model_name = self.metadata.get("model_name", self.model_path.stem)
        self.threshold = float(self.metadata.get("threshold", 0.5))

        model_params = self.metadata.get("params", {})
        self.feature_cols = model_params.get("feature_cols")

        if self.feature_cols is None:
            logger.error("Error al inicializar: Falta 'feature_cols' en los metadatos.")
            raise ValueError(
                "No se han encontrado las columnas del modelo en los metadatos. "
                "El archivo metadata debe contener params.feature_cols."
            )

        logger.info(
            "Predictor configurado con éxito. Modelo: '%s' | Umbral de decisión: %.3f | "
            "Variables requeridas: %d",
            self.model_name,
            self.threshold,
            len(self.feature_cols),
        )

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        """Carga el modelo entrenado desde disco.

        Args:
            model_path (Path): Ruta física del archivo del modelo.

        Returns:
            Any: El estimador/pipeline entrenado cargado.

        Raises:
            FileNotFoundError: Si el archivo no existe en la ruta especificada.
        """
        if not model_path.exists():
            logger.error("No se encontró el archivo del modelo en: %s", model_path)
            raise FileNotFoundError(f"No existe el modelo: {model_path}")

        logger.debug("Cargando modelo binario desde %s...", model_path)
        return joblib.load(model_path)

    @staticmethod
    def _load_metadata(metadata_path: Path) -> dict:
        """Carga los metadatos asociados al modelo desde un archivo JSON.

        Args:
            metadata_path (Path): Ruta física del archivo JSON de metadatos.

        Returns:
            dict: Diccionario con la configuración y parámetros del modelo.

        Raises:
            FileNotFoundError: Si el archivo JSON no existe.
        """
        if not metadata_path.exists():
            logger.error("No se encontró el archivo de metadatos en: %s", metadata_path)
            raise FileNotFoundError(f"No existe el archivo de metadatos: {metadata_path}")

        logger.debug("Cargando archivo de metadatos desde %s...", metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_and_prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valida que los datos contienen todas las variables esperadas por el modelo y las prepara.

        Args:
            data (pd.DataFrame): Datos de entrada en bruto.

        Returns:
            pd.DataFrame: DataFrame filtrado solo con las variables del modelo y tipado flotante.

        Raises:
            ValueError: Si faltan columnas requeridas para ejecutar la inferencia.
        """
        logger.debug("Validando columnas del DataFrame de entrada...")
        missing_cols = [col for col in self.feature_cols if col not in data.columns]

        if missing_cols:
            logger.error("Validación fallida. Columnas faltantes: %s", missing_cols)
            raise ValueError(
                "Faltan columnas necesarias para la predicción: "
                f"{missing_cols}"
            )

        X = data[self.feature_cols].copy()

        # Asegurar formato numérico uniforme
        X = X.astype(float)
        logger.debug("Validación correcta. Matriz de características preparada: %s", X.shape)

        return X

    @staticmethod
    def _risk_level(probability: float, threshold: float) -> str:
        """Asigna un nivel de riesgo cualitativo ("high", "medium", "low") según la probabilidad predicha.

        Args:
            probability (float): Probabilidad estimada de sepsis.
            threshold (float): Umbral de decisión clínico optimizado.

        Returns:
            str: Etiqueta de nivel de riesgo ("high", "medium" o "low").
        """
        if probability >= threshold:
            return "high"
        if probability >= threshold * 0.75:
            return "medium"
        return "low"

    def predict_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Genera predicciones en lote para un DataFrame de registros clínicos.

        Args:
            data (pd.DataFrame): Registros clínicos con las variables del modelo.

        Returns:
            pd.DataFrame: Predicciones con probabilidad, umbral, alerta y nivel de riesgo.
        """
        logger.info("Procesando inferencia para un lote de %d filas...", len(data))
        X = self._validate_and_prepare_input(data)

        probabilities = self.model.predict_proba(X)[:, 1]
        alerts = probabilities >= self.threshold

        results = pd.DataFrame({
            "sepsis_probability": probabilities,
            "threshold": self.threshold,
            "alert": alerts.astype(bool),
            "risk_level": [
                self._risk_level(prob, self.threshold)
                for prob in probabilities
            ],
            "model_name": self.model_name,
        })

        # Mantener identificador de paciente si viene en los datos de entrada
        if "PatientID" in data.columns:
            results.insert(0, "patient_id", data["PatientID"].values)

        if "ICULOS" in data.columns:
            results.insert(
                1 if "patient_id" in results.columns else 0,
                "iculos",
                data["ICULOS"].values,
            )

        logger.info("Predicciones calculadas con éxito. Alertas totales disparadas: %d", int(alerts.sum()))
        return results

    def predict_one(self, record: dict) -> dict:
        """Genera una predicción enriquecida para un único registro clínico en formato diccionario.

        Args:
            record (dict): Diccionario cuyas llaves son las variables de signos vitales/laboratorio.

        Returns:
            dict: Diccionario con la probabilidad, umbral, alerta y nivel de riesgo, mapeados a tipos JSON válidos.
        """
        logger.debug("Generando predicción para un único registro.")
        data = pd.DataFrame([record])
        result = self.predict_dataframe(data).iloc[0].to_dict()

        return {
            key: self._convert_json_value(value)
            for key, value in result.items()
        }

    def predict_many(self, records: list[dict]) -> list[dict]:
        """Genera predicciones enriquecidas para una colección de registros clínicos (lista de dicts).

        Args:
            records (list[dict]): Lista de diccionarios con la información de los pacientes.

        Returns:
            list[dict]: Lista de diccionarios de salida sanitizados para ser compatibles con JSON.
        """
        logger.info("Generando predicciones en lote para una lista de %d diccionarios.", len(records))
        data = pd.DataFrame(records)
        results = self.predict_dataframe(data)

        return [
            {
                key: self._convert_json_value(value)
                for key, value in row.items()
            }
            for row in results.to_dict(orient="records")
        ]

    @staticmethod
    def _convert_json_value(value: Any) -> Any:
        """Convierte tipos de datos nativos de NumPy/Pandas a tipos primitivos estándar de Python.

        Esto previene errores de serialización cuando el resultado se envía a través de APIs JSON
        (por ejemplo, transformando float64 a float, bool_ a bool o valores nulos/NaN a None).

        Args:
            value (Any): Valor bruto extraído del DataFrame resultante.

        Returns:
            Any: Valor convertido y sanitizado para JSON.
        """

        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value


def load_predictor_from_paths(
    model_path: str | Path,
    metadata_path: str | Path,
) -> SepsisPredictor:
    """Carga el motor de inferencia SepsisPredictor especificando rutas explícitas.

    Args:
        model_path (str | Path): Ubicación del modelo empaquetado.
        metadata_path (str | Path): Ubicación del JSON de metadatos.

    Returns:
        SepsisPredictor: Una instancia configurada lista para predecir.
    """
    logger.info("Cargando SepsisPredictor desde rutas explícitas.")
    return SepsisPredictor(
        model_path=model_path,
        metadata_path=metadata_path,
    )


def load_predictor_from_env() -> SepsisPredictor:
    """Carga de manera automática el SepsisPredictor leyendo las rutas desde variables de entorno.

    Por defecto carga el modelo publicado para despliegue simulado en:
        deployment/model/model.joblib
        deployment/model/metadata.json

    Variables opcionales:
        MODEL_PATH: Ruta al archivo binario del modelo.
        METADATA_PATH: Ruta al archivo JSON de metadatos.

    Returns:
        SepsisPredictor: Una instancia del predictor lista para inferencia.
    """
    logger.info("Cargando SepsisPredictor utilizando variables de entorno...")

    model_path = os.getenv(
        "MODEL_PATH",
        "deployment/model/model.joblib",
    )

    metadata_path = os.getenv(
        "METADATA_PATH",
        "deployment/model/metadata.json",
    )

    return SepsisPredictor(
        model_path=model_path,
        metadata_path=metadata_path,
    )