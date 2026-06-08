from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
import shutil
import joblib

import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

from core.plots import (
    plot_roc_pr_curves,
    plot_confusion_matrices_row_patient,
    plot_feature_importance,
    plot_shap_summary,
)
from core.utils import load_params, ensure_dirs, save_json

logger = logging.getLogger(__name__)


def safe_model_name(model_name: str) -> str:
    """Convierte el nombre de un modelo en una cadena segura para nombres de archivos.

    Elimina espacios, paréntesis y caracteres especiales problemáticos, reemplazándolos
    por guiones bajos y convirtiendo todo a minúsculas.

    Args:
        model_name (str): Nombre original del modelo (ej. "Random Forest (v1)").

    Returns:
        str: Nombre del modelo normalizado y seguro para el sistema de archivos.
    """
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def metric_name(metric: str) -> str:
    """Normaliza y estandariza los nombres de las métricas de evaluación.

    Mapea alias comunes de texto (como "auroc" o "F1") a sus nombres canónicos oficiales
    utilizados en la generación de reportes del proyecto.

    Args:
        metric (str): Nombre o alias de la métrica a normalizar.

    Returns:
        str: Nombre estandarizado de la métrica. Si no tiene alias, devuelve la original.
    """
    aliases = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "f1": "F1",
        "F1": "F1",
        "auroc": "AUROC",
        "AUROC": "AUROC",
        "auprc": "AUPRC",
        "AUPRC": "AUPRC",
    }
    return aliases.get(metric, metric)


def read_json(path: str | Path) -> dict:
    """Lee y deserializa un archivo en formato JSON.

    Args:
        path (str | Path): Ruta de acceso al archivo JSON que se desea cargar.

    Returns:
        dict: Contenido del archivo JSON parseado como un diccionario de Python.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics_table(path: str | Path) -> pd.DataFrame:
    """Carga una tabla de métricas en formato CSV generada previamente por train.py.

    Args:
        path (str | Path): Ruta al archivo CSV con las métricas acumuladas.

    Returns:
        pd.DataFrame: DataFrame de pandas indexado por el nombre del modelo.

    Raises:
        FileNotFoundError: Si el archivo de métricas especificado no existe en la ruta.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de métricas: {path}")

    logger.debug("Tabla de métricas guardada en: %s", path)
    return pd.read_csv(path, index_col=0)


def load_threshold_summary(path: str | Path) -> pd.DataFrame:
    """Carga el resumen de los umbrales de decisión óptimos calculados para cada modelo.

    Args:
        path (str | Path): Ruta al archivo CSV que contiene el resumen de umbrales.

    Returns:
        pd.DataFrame: DataFrame con los umbrales óptimos asignados a cada modelo candidato.

    Raises:
        FileNotFoundError: Si el archivo de umbrales especificado no existe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de umbrales: {path}")

    logger.debug("Resumen de umbrales cargado en: %s", path)
    return pd.read_csv(path)


def build_model_comparison(
    df_row: pd.DataFrame,
    df_patient: pd.DataFrame,
    df_thresholds: pd.DataFrame,
    best_metric: str,
) -> pd.DataFrame:
    """Construye una tabla comparativa consolidada ordenando los modelos por rendimiento.

    Combina las métricas calculadas a nivel de fila (observación) y a nivel de paciente,
    añade sus umbrales óptimos asociados y ordena la tabla de mayor a menor según
    la métrica principal de selección especificada.

    Args:
        df_row (pd.DataFrame): Métricas de rendimiento calculadas a nivel de fila.
        df_patient (pd.DataFrame): Métricas de rendimiento calculadas a nivel de paciente.
        df_thresholds (pd.DataFrame): Resumen de umbrales óptimos por modelo.
        best_metric (str): Nombre de la métrica principal para ordenar la comparativa.

    Returns:
        pd.DataFrame: Tabla comparativa consolidada y ordenada de forma descendente.

    Raises:
        ValueError: Si la métrica principal solicitada no se encuentra en las columnas.
    """
    df_comparison = (
        df_row.add_prefix("row_")
        .join(df_patient.add_prefix("patient_"), how="left")
        .reset_index()
        .rename(columns={"index": "model_name"})
    )

    df_comparison = df_comparison.merge(
        df_thresholds,
        on="model_name",
        how="left",
    )

    metric_col = f"row_{best_metric}"
    if metric_col not in df_comparison.columns:
        raise ValueError(
            f"No existe la métrica '{metric_col}' en la tabla comparativa. "
            f"Columnas disponibles: {list(df_comparison.columns)}"
        )

    df_comparison = df_comparison.sort_values(
        metric_col,
        ascending=False,
    ).reset_index(drop=True)

    return df_comparison


def save_latex_comparison_table(
    df_comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Genera y guarda una tabla formateada en LaTeX con la comparativa de modelos.

    Filtra las métricas principales de la tabla comparativa, renombra las columnas
    a un formato legible en español y exporta el código LaTeX listo para ser incluido 
    en reportes académicos o científicos.

    Args:
        df_comparison (pd.DataFrame): Tabla consolidada de comparación de modelos.
        output_path (str | Path): Ruta de destino donde se guardará el archivo `.tex`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "model_name",
        "row_AUROC",
        "row_AUPRC",
        "row_Precision",
        "row_Recall",
        "row_F1",
        "row_Specificity",
        "threshold",
    ]

    cols = [col for col in cols if col in df_comparison.columns]
    df_table = df_comparison[cols].copy()

    rename_cols = {
        "model_name": "Modelo",
        "row_AUROC": "AUROC",
        "row_AUPRC": "AUPRC",
        "row_Precision": "Precision",
        "row_Recall": "Recall",
        "row_F1": "F1-score",
        "row_Specificity": "Especificidad",
        "threshold": "Umbral",
    }
    df_table = df_table.rename(columns=rename_cols)

    latex = df_table.to_latex(
        index=False,
        float_format="%.4f",
        caption="Comparativa de rendimiento de los modelos evaluados.",
        label="tab:comparativa_modelos",
        escape=True,
    )

    output_path.write_text(latex, encoding="utf-8")


def load_prediction_results(predictions_dir: str | Path) -> dict:
    """Carga las predicciones de test de todos los modelos para las curvas ROC y PR.

    Escanea el directorio especificado en busca de archivos Parquet de predicciones, 
    extrayendo las probabilidades asignadas y las etiquetas reales correspondientes 
    de cada estimador para su posterior graficación masiva.

    Args:
        predictions_dir (str | Path): Directorio que almacena los archivos de predicciones.

    Returns:
        dict: Diccionario estructurado donde las llaves son los nombres de los modelos y los 
            valores son diccionarios con arreglos NumPy para 'y_true' e 'y_prob'.

    Raises:
        FileNotFoundError: Si no se encuentran archivos de predicción en el directorio.
        RuntimeError: Si ningún archivo Parquet cargado es válido o contiene las columnas requeridas.
    """
    predictions_dir = Path(predictions_dir)
    prediction_files = sorted(predictions_dir.glob("*_test_predictions.parquet"))

    if not prediction_files:
        raise FileNotFoundError(
            f"No se han encontrado predicciones de test en: {predictions_dir}"
        )

    results = {}
    for pred_path in prediction_files:
        df_pred = pd.read_parquet(pred_path)
        required_cols = {"y_true", "y_prob", "model_name"}

        if not required_cols.issubset(df_pred.columns):
            logger.warning("Se omite %s: faltan columnas requeridas.", pred_path.name)
            continue

        model_name = str(df_pred["model_name"].iloc[0])
        results[model_name] = {
            "y_true": df_pred["y_true"].to_numpy(),
            "y_prob": df_pred["y_prob"].to_numpy(),
        }

    if not results:
        raise RuntimeError("No se han podido cargar predicciones válidas.")

    return results


def get_best_model_name(
    df_comparison: pd.DataFrame,
    best_metric: str,
    metric_level: str = "row",
) -> str:
    """Identifica el nombre del mejor modelo basándose en una métrica y nivel específicos.

    Args:
        df_comparison (pd.DataFrame): Tabla comparativa con los resultados consolidados.
        best_metric (str): Nombre de la métrica objetivo a maximizar (ej. "AUPRC").
        metric_level (str, opcional): Nivel de resolución de la métrica ("row" o "patient"). 
            Por defecto es "row".

    Returns:
        str: Nombre de cadena exacto del modelo con el mejor desempeño bajo la métrica dada.

    Raises:
        ValueError: Si la combinación de nivel y métrica no genera una columna existente.
    """
    metric_col = f"{metric_level}_{best_metric}"
    if metric_col not in df_comparison.columns:
        raise ValueError(
            f"No existe la columna '{metric_col}'. "
            f"Columnas disponibles: {list(df_comparison.columns)}"
        )

    best_idx = df_comparison[metric_col].idxmax()
    return str(df_comparison.loc[best_idx, "model_name"])


def load_test_features_for_model(
    params: dict,
    metadata: dict,
    model_name: str,
) -> pd.DataFrame:
    """Carga el conjunto de características (features) de test adecuado para el modelo.

    Determina automáticamente si el modelo requiere variables transformadas/escaladas
    (como la Regresión Logística) o variables en su formato original (como los modelos 
    basados en árboles) basándose en su nombre y en los metadatos de variables guardados.

    Args:
        params (dict): Diccionario global de configuración de parámetros (`params.yaml`).
        metadata (dict): Metadatos específicos del modelo recuperados del archivo JSON asociado.
        model_name (str): Nombre comercial/identificador del modelo que se está evaluando.

    Returns:
        pd.DataFrame: Conjunto de datos de test filtrado únicamente con las variables predictoras exactas.

    Raises:
        ValueError: Si no se localiza ningún conjunto de datos con las columnas esperadas por el modelo.
    """
    data_cfg = params.get("data", {})
    target_cfg = params.get("target", {})

    target_col = target_cfg.get("name", "SepsisLabel")
    aux_cols = target_cfg.get("aux_cols", ["PatientID", "Hospital", "TimeStep"])

    model_params = metadata.get("params", {})
    feature_cols = model_params.get("feature_cols")

    if feature_cols is None:
        test_path = Path(data_cfg.get("test_path", "data/processed/test_preprocessed.parquet"))
        df_test = pd.read_parquet(test_path)
        excluded = set(aux_cols + [target_col])
        feature_cols = [col for col in df_test.columns if col not in excluded]
        return df_test[feature_cols].copy()

    test_path = Path(data_cfg.get("test_path", "data/processed/test_preprocessed.parquet"))
    test_scaled_path = Path(
        data_cfg.get("test_scaled_path", "data/processed/test_preprocessed_scaled.parquet")
    )

    use_scaled = "logistic" in model_name.lower()
    candidate_paths = [test_scaled_path, test_path] if use_scaled else [test_path, test_scaled_path]

    for path in candidate_paths:
        if not path.exists():
            continue
        df_test = pd.read_parquet(path)
        if all(col in df_test.columns for col in feature_cols):
            return df_test[feature_cols].copy()

    raise ValueError(
        "No se ha encontrado ningún conjunto de test que contenga todas las "
        f"variables del modelo '{model_name}'."
    )

def promote_model_for_serving(
    best_model_name: str,
    model_path: Path,
    metadata_path: Path,
    serving_dir: Path,
    best_metric: str,
    metric_level: str,
    best_threshold: float,
) -> None:
    """Copia el mejor modelo a una ruta estable para despliegue simulado.

    La API siempre cargará:
        deployment/model/model.joblib
        deployment/model/metadata.json

    De este modo, la API no depende del nombre concreto del algoritmo ganador.
    """
    serving_dir.mkdir(parents=True, exist_ok=True)

    serving_model_path = serving_dir / "model.joblib"
    serving_metadata_path = serving_dir / "metadata.json"
    serving_config_path = serving_dir / "serving_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No se encuentra el modelo final: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encuentran los metadatos del modelo final: {metadata_path}"
        )

    shutil.copy2(model_path, serving_model_path)
    shutil.copy2(metadata_path, serving_metadata_path)

    save_json(
        {
            "final_model_name": best_model_name,
            "source_model_path": str(model_path),
            "source_metadata_path": str(metadata_path),
            "serving_model_path": str(serving_model_path),
            "serving_metadata_path": str(serving_metadata_path),
            "selection_metric": best_metric,
            "selection_level": metric_level,
            "threshold": float(best_threshold),
        },
        serving_config_path,
    )

    logger.info("Modelo final promocionado para inferencia en: %s", serving_dir)
    logger.info("Modelo publicado: %s", serving_model_path)
    logger.info("Metadatos publicados: %s", serving_metadata_path)

def register_final_model_mlflow(
    model,
    best_model_name: str,
    model_path: Path,
    metadata_path: Path,
    serving_config_path: Path,
    params: dict,
    metrics: dict,
) -> None:
    """Registra el modelo final en MLflow Model Registry."""

    mlflow_cfg = params.get("mlflow", {})
    enabled = bool(mlflow_cfg.get("enabled", False))
    register_model = bool(mlflow_cfg.get("register_final_model", True))

    if not enabled or not register_model:
        return

    if mlflow is None:
        raise ImportError("MLflow está activado, pero no está instalado.")

    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "sepsis-mlops")
    registered_model_name = mlflow_cfg.get(
        "registered_model_name",
        "sepsis-final-model",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"register_final_model_{best_model_name}"):
        mlflow.set_tag("stage", "model_registration")
        mlflow.set_tag("final_model_name", best_model_name)

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(model_path), artifact_path="deployment_artifacts")
        mlflow.log_artifact(str(metadata_path), artifact_path="deployment_artifacts")

        if serving_config_path.exists():
            mlflow.log_artifact(str(serving_config_path), artifact_path="deployment_artifacts")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
        )

    logger.info(
        "Modelo final registrado en MLflow Registry como: %s",
        registered_model_name,
    )

def main(params: dict) -> None:
    """Flujo principal de evaluación final de modelos y generación de artefactos.

    Inicializa los parámetros, consolida las tablas de métricas a nivel fila/paciente,
    selecciona de manera automatizada el mejor estimador del experimento, calcula las 
    matrices de confusión optimizadas por umbral, y opcionalmente construye los gráficos 
    explicativos de importancia de variables globales y análisis de valores SHAP locales.

    Args:
        params_path (str): Ruta al archivo de configuración centralizado (`params.yaml`).
    """

    outputs_cfg = params.get("outputs", {})
    target_cfg = params.get("target", {})
    evaluation_cfg = params.get("evaluation", {})

    target_col = target_cfg.get("name", "SepsisLabel")
    patient_col = target_cfg.get("patient_col", "PatientID")

    models_dir = Path(outputs_cfg.get("models_dir", "models"))
    train_metrics_dir = Path(outputs_cfg.get("train_metrics_dir", "reports/metrics/train"))
    eval_metrics_dir = Path(outputs_cfg.get("eval_metrics_dir", "reports/metrics/evaluate"))
    predictions_dir = Path(outputs_cfg.get("predictions_dir", "reports/predictions"))
    figures_dir = Path(outputs_cfg.get("figures_dir", "reports/figures"))
    tables_dir = Path(outputs_cfg.get("tables_dir", "reports/tables"))
    serving_dir = Path(outputs_cfg.get("serving_dir", "deployment/model"))

    ensure_dirs([
        models_dir,
        train_metrics_dir,
        eval_metrics_dir,
        predictions_dir,
        figures_dir,
        tables_dir,
        serving_dir
    ])

    best_metric = metric_name(evaluation_cfg.get("best_model_metric", "AUPRC"))
    metric_level = evaluation_cfg.get("best_model_level", "row")
    shap_sample_size = int(evaluation_cfg.get("shap_sample_size", 2000))
    shap_max_display = int(evaluation_cfg.get("shap_max_display", 20))
    top_n_features = int(evaluation_cfg.get("top_n_features", 20))
    generate_shap = bool(evaluation_cfg.get("generate_shap", True))
    generate_feature_importance = bool(
        evaluation_cfg.get("generate_feature_importance", True)
    )

    logger.info("Cargando métricas generadas por train.py...")

    df_row = load_metrics_table(train_metrics_dir / "all_model_metrics_row.csv")
    df_patient = load_metrics_table(train_metrics_dir / "all_model_metrics_patient.csv")
    df_thresholds = load_threshold_summary(train_metrics_dir / "threshold_summary.csv")

    df_comparison = build_model_comparison(
        df_row=df_row,
        df_patient=df_patient,
        df_thresholds=df_thresholds,
        best_metric=best_metric,
    )

    comparison_path = eval_metrics_dir / "model_comparison.csv"
    df_comparison.to_csv(comparison_path, index=False)

    save_latex_comparison_table(
        df_comparison=df_comparison,
        output_path=tables_dir / "tabla_comparativa.tex",
    )

    best_model_name = get_best_model_name(
        df_comparison=df_comparison,
        best_metric=best_metric,
        metric_level=metric_level,
    )
    safe_name = safe_model_name(best_model_name)

    logger.info("Mejor modelo seleccionado: %s", best_model_name)

    best_row = df_comparison[df_comparison["model_name"] == best_model_name].iloc[0]
    best_threshold = float(best_row["threshold"])

    best_model_info = {
        "best_model_name": best_model_name,
        "selection_metric": best_metric,
        "selection_level": metric_level,
        "threshold": best_threshold,
        "metrics": best_row.to_dict(),
    }

    save_json(
        best_model_info,
        eval_metrics_dir / "best_model_metrics.json",
    )

    logger.info("Generando curvas ROC y Precision-Recall...")
    prediction_results = load_prediction_results(predictions_dir)

    plot_roc_pr_curves(
        results=prediction_results,
        output_path=figures_dir / "all_models_roc_pr.pdf",
    )
    plot_roc_pr_curves(
        results=prediction_results,
        output_path=figures_dir / "all_models_roc_pr.png",
    )

    logger.info("Generando matrices de confusión del mejor modelo...")
    best_predictions_path = predictions_dir / f"{safe_name}_test_predictions.parquet"

    if not best_predictions_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de predicciones del mejor modelo: "
            f"{best_predictions_path}"
        )

    df_pred_best = pd.read_parquet(best_predictions_path)
    df_ref = df_pred_best.copy()
    df_ref[target_col] = df_ref["y_true"]

    plot_confusion_matrices_row_patient(
        y_true_row=df_pred_best["y_true"].to_numpy(),
        y_prob_row=df_pred_best["y_prob"].to_numpy(),
        df_ref=df_ref,
        threshold=best_threshold,
        output_path=figures_dir / "confusion_matrix_row_patient_normalized.pdf",
        target_col=target_col,
        patient_col=patient_col,
    )
    plot_confusion_matrices_row_patient(
        y_true_row=df_pred_best["y_true"].to_numpy(),
        y_prob_row=df_pred_best["y_prob"].to_numpy(),
        df_ref=df_ref,
        threshold=best_threshold,
        output_path=figures_dir / "confusion_matrix_row_patient_normalized.png",
        target_col=target_col,
        patient_col=patient_col,
    )

    model_path = models_dir / f"{safe_name}.joblib"
    metadata_path = models_dir / f"{safe_name}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No se encuentra el modelo guardado: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encuentran los metadatos: {metadata_path}")

    promote_model_for_serving(
        best_model_name=best_model_name,
        model_path=model_path,
        metadata_path=metadata_path,
        serving_dir=serving_dir,
        best_metric=best_metric,
        metric_level=metric_level,
        best_threshold=best_threshold
    )


    logger.info("Cargando modelo final...")
    best_model = joblib.load(model_path)
    metadata = read_json(metadata_path)

    # Versionado del modelo final
    final_metrics_for_registry = {
        "row_AUPRC": float(best_row.get("row_AUPRC", np.nan)),
        "row_AUROC": float(best_row.get("row_AUROC", np.nan)),
        "row_F1": float(best_row.get("row_F1", np.nan)),
        "row_Recall": float(best_row.get("row_Recall", np.nan)),
        "row_Precision": float(best_row.get("row_Precision", np.nan)),
        "patient_AUPRC": float(best_row.get("patient_AUPRC", np.nan)),
        "patient_AUROC": float(best_row.get("patient_AUROC", np.nan)),
        "patient_F1": float(best_row.get("patient_F1", np.nan)),
        "threshold": float(best_threshold),
    }

    serving_config_path = serving_dir / "serving_config.json"

    register_final_model_mlflow(
        model=best_model,
        best_model_name=best_model_name,
        model_path=model_path,
        metadata_path=metadata_path,
        serving_config_path=serving_config_path,
        params=params,
        metrics=final_metrics_for_registry,
    )

    feature_cols = metadata.get("params", {}).get("feature_cols")

    if feature_cols is None:
        logger.warning("No se han encontrado feature_cols en los metadatos.")
        logger.warning("Se omiten importancia de variables y SHAP.")
    else:
        X_test_model = load_test_features_for_model(
            params=params,
            metadata=metadata,
            model_name=best_model_name,
        )

        if generate_feature_importance:
            logger.info("Generando importancia de variables...")
            try:
                plot_feature_importance(
                    model=best_model,
                    feature_names=feature_cols,
                    output_path=figures_dir / "feature_importance_final_model.pdf",
                    top_n=top_n_features,
                    title=f"Importancia de variables — {best_model_name}",
                )
                plot_feature_importance(
                    model=best_model,
                    feature_names=feature_cols,
                    output_path=figures_dir / "feature_importance_final_model.png",
                    top_n=top_n_features,
                    title=f"Importancia de variables — {best_model_name}",
                )
            except Exception as e:
                logger.error("No se pudo generar la importancia de variables: %s", e)

        if generate_shap:
            logger.info("Generando gráfico SHAP...")
            try:
                plot_shap_summary(
                    model=best_model,
                    X=X_test_model,
                    output_path_png=figures_dir / "shap_summary_final_model.png",
                    output_path_pdf=figures_dir / "shap_summary_final_model.pdf",
                    feature_names=feature_cols,
                    sample_size=shap_sample_size,
                    random_state=int(params.get("random_state", 42)),
                    max_display=shap_max_display,
                    title=f"SHAP Summary Plot — {best_model_name}",
                )
            except Exception as e:
                logger.error("No se pudo generar SHAP para este modelo: %s", e)

    logger.info("=" * 80)
    logger.info("Evaluación finalizada correctamente.")
    logger.info("=" * 80)
    logger.info("Comparativa de modelos: %s", comparison_path)
    logger.info("Mejor modelo: %s", best_model_name)
    logger.info("Umbral: %.3f", best_threshold)
    logger.info("Figuras guardadas en: %s", figures_dir)
    logger.info("Tablas guardadas en: %s", tables_dir)
    logger.info("Métricas de entrenamiento leídas de: %s", train_metrics_dir)
    logger.info("Métricas de evaluación guardadas en: %s", eval_metrics_dir)
    logger.info("Artefactos de despliegue guardados en: %s", serving_dir)


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evalúa los modelos entrenados y genera artefactos finales."
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="Ruta al archivo params.yaml.",
    )
    args = parser.parse_args()

    params=load_params(args.params)

    log_level = params.get("logging", {}).get("level", "INFO").upper()
    if log_level == "NONE":
        logging.disable(logging.CRITICAL)
    else:
        logging.getLogger().setLevel(log_level)

    logger.info("Nivel de logging configurado: %s", log_level)
    logger.debug("params.yaml cargado desde: %s", args.params)


    
    main(params)