from __future__ import annotations

import argparse
import time
from pathlib import Path
import contextlib
import logging

import numpy as np
import pandas as pd


from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import mlflow
except ImportError:
    mlflow = None

from core.metrics import (
    compute_metrics_row,
    compute_metrics_patient,
    find_best_threshold_row,
)
from core.utils import (
    load_params,
    ensure_dirs,
    save_json,
    save_model_artifacts,
)

logger = logging.getLogger(__name__)


def safe_model_name(model_name: str) -> str:
    """Convierte el nombre de un modelo en un nombre seguro para archivos."""
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def metric_name(metric: str) -> str:
    """Normaliza nombres de métricas para que coincidan con las claves devueltas."""
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


def build_threshold_grid(training_cfg: dict) -> np.ndarray:
    """Construye la rejilla de umbrales candidatos."""
    start = float(training_cfg.get("thresholds_start", 0.05))
    end = float(training_cfg.get("thresholds_end", 0.95))
    step = float(training_cfg.get("thresholds_step", 0.01))

    grid = np.round(np.arange(start, end + step / 2, step), 10)

    logger.debug(
        "Rejilla de umbrales: start=%.2f, end=%.2f, step=%.2f → %d candidatos",
        start, end, step, len(grid),
    )

    return grid


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    aux_cols: list[str],
) -> list[str]:
    """Devuelve las columnas predictoras excluyendo target y columnas auxiliares."""
    excluded = set(aux_cols + [target_col])
    return [col for col in df.columns if col not in excluded]


def prepare_search_subset(
    df_train: pd.DataFrame,
    df_train_scaled: pd.DataFrame,
    feature_cols: list[str],
    feature_cols_scaled: list[str],
    target_col: str,
    patient_col: str,
    neg_pos_ratio: int,
    random_state: int,
) -> dict:
    """Crea una submuestra por pacientes para búsqueda de hiperparámetros.

    Incluye todos los pacientes positivos y una muestra de negativos.
    """

    patient_labels = (
        df_train[[patient_col, target_col]]
        .groupby(patient_col)[target_col]
        .max()
        .reset_index()
    )

    pos_patients = patient_labels[patient_labels[target_col] == 1]
    neg_patients = patient_labels[patient_labels[target_col] == 0]

    if len(pos_patients) == 0:
        raise ValueError("No hay pacientes positivos en train. No se puede crear la submuestra.")

    sample_neg = neg_patients.sample(
        n=min(len(neg_patients), len(pos_patients) * neg_pos_ratio),
        random_state=random_state,
    )

    selected_patients = pd.concat([pos_patients, sample_neg])[patient_col]

    df_search = df_train[df_train[patient_col].isin(selected_patients)].copy()
    df_search_scaled = df_train_scaled[
        df_train_scaled[patient_col].isin(selected_patients)
    ].copy()

    logger.debug(
        "Pacientes positivos: %d | Negativos muestreados: %d | Total submuestra: %d",
        len(pos_patients),
        len(sample_neg),
        len(selected_patients),
    )

    return {
        "X_search": df_search[feature_cols].reset_index(drop=True),
        "y_search": df_search[target_col].reset_index(drop=True),
        "groups_search": df_search[patient_col].reset_index(drop=True),
        "X_search_lr": df_search_scaled[feature_cols_scaled].reset_index(drop=True),
        "y_search_lr": df_search_scaled[target_col].reset_index(drop=True),
        "groups_search_lr": df_search_scaled[patient_col].reset_index(drop=True),
        "selected_patients": selected_patients,
    }


def save_predictions(
    df_ref: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split: str,
    output_path: str | Path,
    target_col: str,
    patient_col: str,
) -> None:
    """Guarda predicciones a nivel de fila."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_cols = [c for c in [patient_col, "TimeStep", "ICULOS", "Hospital"] if c in df_ref.columns]

    df_preds = df_ref[meta_cols + [target_col]].copy()
    df_preds = df_preds.rename(columns={target_col: "y_true"})

    df_preds["y_prob"] = np.asarray(y_prob)
    df_preds["y_pred"] = (df_preds["y_prob"] >= threshold).astype(int)
    df_preds["model_name"] = model_name
    df_preds["split"] = split

    df_preds.to_parquet(output_path, index=False)


def log_mlflow_run(
    enabled: bool,
    model_name: str,
    model_params: dict,
    threshold: float,
    metrics_row: dict,
    metrics_patient: dict,
    artifacts: list[Path],
) -> None:
    """Registra parámetros, métricas y artefactos en el run de MLflow activo."""
    if not enabled:
        return

    if mlflow is None:
        raise ImportError("MLflow está activado en params.yaml, pero no está instalado.")

    for key, value in model_params.items():
        if isinstance(value, (list, dict, tuple)):
            mlflow.log_param(key, str(value))
        else:
            mlflow.log_param(key, value)

    mlflow.log_param("threshold", float(threshold))

    for key, value in metrics_row.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            mlflow.log_metric(f"row_{key}", float(value))

    for key, value in metrics_patient.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            mlflow.log_metric(f"patient_{key}", float(value))

    for artifact in artifacts:
        artifact = Path(artifact)
        if artifact.exists():
            mlflow.log_artifact(str(artifact))


def fit_evaluate_and_save(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_train_ref: pd.DataFrame,
    df_test_ref: pd.DataFrame,
    threshold_metric: str,
    thresholds_grid: np.ndarray,
    n_splits: int,
    output_dirs: dict,
    target_col: str,
    patient_col: str,
    model_params: dict | None = None,
    tune_threshold: bool = True,
    fixed_threshold: float = 0.5,
    mlflow_enabled: bool = False,
) -> dict:
    """Entrena un modelo, ajusta umbral con OOF, evalúa en test y guarda artefactos."""

    if model_params is None:
        model_params = {}

    safe_name = safe_model_name(model_name)

    logger.info("=" * 80)
    logger.info("Entrenando modelo: %s", model_name)
    logger.info("=" * 80)

    start_time = time.time()
    model.fit(X_train, y_train)
    retrain_time = time.time() - start_time

    logger.debug("Features usadas (%d): %s", X_train.shape[1], list(X_train.columns))

    logger.info("Reentrenamiento con train completo: %.2f segundos", retrain_time)
    if tune_threshold:
        logger.info("Calculando predicciones out-of-fold para ajuste de umbral...")

        y_prob_oof = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=GroupKFold(n_splits=n_splits),
            groups=groups_train,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

    
        best_threshold, threshold_scores = find_best_threshold_row(
            y_true=y_train,
            y_prob=y_prob_oof,
            metric=threshold_metric,
            thresholds=thresholds_grid,
        )
    else:
        best_threshold = fixed_threshold
        threshold_scores = pd.DataFrame()
        logger.debug("Umbral fijo aplicado: %.3f (sin ajuste OOF)", fixed_threshold)
        y_prob_oof = model.predict_proba(X_train)[:, 1]

    best_threshold = float(best_threshold)

    logger.info("Umbral seleccionado para %s: %.3f", model_name, best_threshold)

    y_prob_test = model.predict_proba(X_test)[:, 1]

    metrics_row = compute_metrics_row(
        y_true=y_test,
        y_prob=y_prob_test,
        threshold=best_threshold,
    )

    metrics_patient = compute_metrics_patient(
        df_ref=df_test_ref,
        y_prob=y_prob_test,
        threshold=best_threshold,
        target_col=target_col,
        patient_col=patient_col,
    )

    logger.debug("Guardando artefactos para %s en %s", model_name, output_dirs["models_dir"])

    model_params_to_save = {
        **model_params,
        "feature_cols": list(X_train.columns),
        "n_features": int(X_train.shape[1]),
        "threshold_metric": threshold_metric,
        "retrain_time_seconds": float(retrain_time),
    }

    save_model_artifacts(
        model=model,
        model_name=model_name,
        threshold=best_threshold,
        metrics_row=metrics_row,
        metrics_patient=metrics_patient,
        output_dirs=output_dirs,
        params=model_params_to_save,
    )

    model_path = Path(output_dirs["models_dir"]) / f"{safe_name}.joblib"
    metadata_path = Path(output_dirs["models_dir"]) / f"{safe_name}_metadata.json"

    metrics_json_path = Path(output_dirs["metrics_dir"]) / f"{safe_name}_metrics.json"
    save_json(
        {
            "model_name": model_name,
            "threshold": best_threshold,
            "metrics_row": metrics_row,
            "metrics_patient": metrics_patient,
            "params": model_params_to_save,
        },
        metrics_json_path,
    )

    oof_predictions_path = Path(output_dirs["predictions_dir"]) / f"{safe_name}_oof_predictions.parquet"
    test_predictions_path = Path(output_dirs["predictions_dir"]) / f"{safe_name}_test_predictions.parquet"

    save_predictions(
        df_ref=df_train_ref,
        y_prob=y_prob_oof,
        threshold=best_threshold,
        model_name=model_name,
        split="oof",
        output_path=oof_predictions_path,
        target_col=target_col,
        patient_col=patient_col,
    )

    save_predictions(
        df_ref=df_test_ref,
        y_prob=y_prob_test,
        threshold=best_threshold,
        model_name=model_name,
        split="test",
        output_path=test_predictions_path,
        target_col=target_col,
        patient_col=patient_col,
    )

    threshold_scores_path = Path(output_dirs["metrics_dir"]) / f"{safe_name}_threshold_scores.csv"
    if not threshold_scores.empty:
        threshold_scores.to_csv(threshold_scores_path, index=False)
    else:
        logger.debug("No hay threshold_scores que guardar para %s", model_name)

    log_mlflow_run(
        enabled=mlflow_enabled,
        model_name=model_name,
        model_params=model_params_to_save,
        threshold=best_threshold,
        metrics_row=metrics_row,
        metrics_patient=metrics_patient,
        artifacts=[
            model_path,
            metadata_path,
            metrics_json_path,
            oof_predictions_path,
            test_predictions_path,
            threshold_scores_path,
        ],
    )

    logger.info("Métricas a nivel fila:")
    for key, value in metrics_row.items():
        logger.info("  %s: %s", key, value)

    logger.info("Métricas a nivel paciente:")
    for key, value in metrics_patient.items():
        logger.info("  %s: %s", key, value)

    return {
        "model_name": model_name,
        "model": model,
        "threshold": best_threshold,
        "metrics_row": metrics_row,
        "metrics_patient": metrics_patient,
        "y_prob_test": y_prob_test,
        "retrain_time_seconds": retrain_time,
        "model_params": model_params_to_save,
    }


def run_randomized_search(
    model_name: str,
    estimator,
    param_distributions: dict,
    X_search: pd.DataFrame,
    y_search: pd.Series,
    groups_search: pd.Series,
    n_iter: int,
    n_splits: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
    verbose: int,
    output_dirs: dict,
) -> tuple[dict, float, float]:
    """Ejecuta RandomizedSearchCV y devuelve mejores parámetros, score y tiempo."""

    logger.info("-" * 80)
    logger.info("Búsqueda de hiperparámetros: %s", model_name)
    logger.info("-" * 80)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=GroupKFold(n_splits=n_splits),
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise",
    )

    logger.debug(
        "Iniciando búsqueda: n_iter=%d, n_splits=%d, scoring=%s, n_jobs=%d",
        n_iter, n_splits, scoring, n_jobs,
    )

    start_time = time.time()
    search.fit(X_search, y_search, groups=groups_search)
    search_time = time.time() - start_time

    safe_name = safe_model_name(model_name)
    cv_results_path = Path(output_dirs["metrics_dir"]) / f"{safe_name}_search_cv_results.csv"
    logger.debug("CV results guardados en: %s", cv_results_path)
    pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)

    logger.info("Tiempo de búsqueda: %.2f segundos", search_time)
    logger.info("Mejores parámetros: %s", search.best_params_)
    logger.info("Mejor %s: %.4f", scoring, search.best_score_)

    return search.best_params_, float(search.best_score_), float(search_time)

def build_model_configs(
    models_cfg: dict,
    random_state: int,
    estimator_n_jobs: int,
    model_n_jobs: int,
    search_n_jobs: int,
    search_data: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    X_train_lr: pd.DataFrame,
    y_train_lr: pd.Series,
    groups_train_lr: pd.Series,
    X_test_lr: pd.DataFrame,
    y_test_lr: pd.Series,
    df_train_scaled: pd.DataFrame,
    df_test_scaled: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict]:
    """Construye la lista de configuraciones de modelos a entrenar.

    Cada entrada define el estimador de búsqueda, la distribución de
    hiperparámetros, el estimador final y los datos de entrenamiento y
    evaluación asociados.

    Args:
        random_state (int): Semilla de aleatoriedad global.
        estimator_n_jobs (int): Paralelismo interno del estimador durante
        la búsqueda de hiperparámetros.
        model_n_jobs (int): Paralelismo interno del estimador final.
        search_n_jobs (int): Paralelismo del ajuste de hiperparámetros.
        search_data (dict): Submuestra para búsqueda de hiperparámetros,
        generada por prepare_search_subset.
        X_train (pd.DataFrame): Features de entrenamiento sin escalar.
        y_train (pd.Series): Target de entrenamiento.
        groups_train (pd.Series): Grupos de pacientes para GroupKFold.
        X_test (pd.DataFrame): Features de test sin escalar.
        y_test (pd.Series): Target de test.
        df_train (pd.DataFrame): DataFrame de referencia de train sin escalar.
        df_test (pd.DataFrame): DataFrame de referencia de test sin escalar.
        X_train_lr (pd.DataFrame): Features de entrenamiento escaladas.
        y_train_lr (pd.Series): Target de entrenamiento para datos escalados.
        groups_train_lr (pd.Series): Grupos de pacientes para datos escalados.
        X_test_lr (pd.DataFrame): Features de test escaladas.
        y_test_lr (pd.Series): Target de test para datos escalados.
        df_train_scaled (pd.DataFrame): DataFrame de referencia de train escalado.
        df_test_scaled (pd.DataFrame): DataFrame de referencia de test escalado.
        feature_cols (list[str]): Columnas predictoras sin escalar.

    Returns:
        list[dict]: Lista de configuraciones de modelos. Cada entrada contiene:
        "name", "search_estimator", "param_distributions", "search_X",
        "search_y", "search_groups", "search_n_jobs", "final_estimator",
        "X_train", "y_train", "groups_train", "X_test", "y_test",
        "df_train_ref", "df_test_ref" y "extra_params".
    """
    configs = []

    scale_pos_weight_search = (
        (search_data["y_search"] == 0).sum()
        / max((search_data["y_search"] == 1).sum(), 1)
    )
    scale_pos_weight_train = (
        (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    )

    logger.debug(
        "scale_pos_weight — búsqueda: %.2f | entrenamiento: %.2f",
        scale_pos_weight_search,
        scale_pos_weight_train,
    )

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    if models_cfg.get("logistic_regression", True): # siempre disponible
        configs.append({
            "name": "Logistic Regression",
            "search_estimator": Pipeline([
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    tol=1e-3,
                    random_state=random_state,
                    solver="saga",
                    n_jobs=estimator_n_jobs,
                )),
            ]),
            "param_distributions": {
                "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l1", "l2"],
            },
            "search_X": search_data["X_search_lr"],
            "search_y": search_data["y_search_lr"],
            "search_groups": search_data["groups_search_lr"],
            "search_n_jobs": search_n_jobs,
            "final_estimator": lambda best: Pipeline([
                ("clf", LogisticRegression(
                    **{k.replace("clf__", ""): v for k, v in best.items()},
                    class_weight="balanced",
                    max_iter=500,
                    tol=1e-3,
                    random_state=random_state,
                    solver="saga",
                    n_jobs=model_n_jobs,
                )),
            ]),
            "X_train": X_train_lr,
            "y_train": y_train_lr,
            "groups_train": groups_train_lr,
            "X_test": X_test_lr,
            "y_test": y_test_lr,
            "df_train_ref": df_train_scaled,
            "df_test_ref": df_test_scaled,
            "extra_params": {},
        })

        logger.debug("Configuración añadida: Logistic Regression")

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    if models_cfg.get("random_forest", True):
        configs.append({
            "name": "Random Forest",
            "search_estimator": Pipeline([
                ("clf", RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=estimator_n_jobs,
                )),
            ]),
            "param_distributions": {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [8, 10, 20],
                "clf__min_samples_split": [2, 10, 20],
                "clf__min_samples_leaf": [1, 4, 10],
                "clf__max_features": ["sqrt"],
            },
            "search_X": search_data["X_search"],
            "search_y": search_data["y_search"],
            "search_groups": search_data["groups_search"],
            "search_n_jobs": search_n_jobs,
            "final_estimator": lambda best: Pipeline([
                ("clf", RandomForestClassifier(
                    **{k.replace("clf__", ""): v for k, v in best.items()},
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=model_n_jobs,
                )),
            ]),
            "X_train": X_train,
            "y_train": y_train,
            "groups_train": groups_train,
            "X_test": X_test,
            "y_test": y_test,
            "df_train_ref": df_train,
            "df_test_ref": df_test,
            "extra_params": {},
        })

        logger.debug("Configuración añadida: Random Forest")

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    if models_cfg.get("xgboost", True):
        if xgb is None:
            raise ImportError("XGBoost está activado, pero no está instalado.")
        configs.append({
            "name": "XGBoost",
            "search_estimator": Pipeline([
                ("clf", xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight_search,
                    eval_metric="aucpr",
                    random_state=random_state,
                    n_jobs=estimator_n_jobs,
                    verbosity=0,
                    tree_method="hist",
                )),
            ]),
            "param_distributions": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.03, 0.05, 0.07, 0.1],
                "clf__subsample": [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 1.0],
                "clf__reg_alpha": [0, 0.1, 0.5],
                "clf__reg_lambda": [5, 10, 15],
            },
            "search_X": search_data["X_search"],
            "search_y": search_data["y_search"],
            "search_groups": search_data["groups_search"],
            "search_n_jobs": search_n_jobs,
            "final_estimator": lambda best: Pipeline([
                ("clf", xgb.XGBClassifier(
                    **{k.replace("clf__", ""): v for k, v in best.items()},
                    scale_pos_weight=scale_pos_weight_train,
                    eval_metric="aucpr",
                    random_state=random_state,
                    n_jobs=model_n_jobs,
                    verbosity=0,
                    tree_method="hist",
                )),
            ]),
            "X_train": X_train,
            "y_train": y_train,
            "groups_train": groups_train,
            "X_test": X_test,
            "y_test": y_test,
            "df_train_ref": df_train,
            "df_test_ref": df_test,
            "extra_params": {"scale_pos_weight": float(scale_pos_weight_train)},
        })

        logger.debug("Configuración añadida: XGBoost")

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    
    if models_cfg.get("lightgbm", True) or models_cfg.get("lightgbm_without_hospadmtime", True):
        if lgb is None:
            raise ImportError("LightGBM está activado, pero no está instalado.")

        lgbm_search_estimator = Pipeline([
            ("clf", lgb.LGBMClassifier(
                is_unbalance=True,
                random_state=random_state,
                n_jobs=estimator_n_jobs,
                verbose=-1,
            )),
        ])
        lgbm_param_dist = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [-1, 5, 10],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__num_leaves": [31, 63, 127],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0],
            "clf__reg_alpha": [0, 0.1, 0.5],
            "clf__reg_lambda": [1, 5, 10],
        }

        def make_lgbm_final(best):
            return Pipeline([
                ("clf", lgb.LGBMClassifier(
                    **{k.replace("clf__", ""): v for k, v in best.items()},
                    is_unbalance=True,
                    random_state=random_state,
                    n_jobs=model_n_jobs,
                    verbose=-1,
                )),
            ])

        if models_cfg.get("lightgbm", True):
            configs.append({
                "name": "LightGBM",
                "search_estimator": lgbm_search_estimator,
                "param_distributions": lgbm_param_dist,
                "search_X": search_data["X_search"],
                "search_y": search_data["y_search"],
                "search_groups": search_data["groups_search"],
                "search_n_jobs": search_n_jobs,
                "final_estimator": make_lgbm_final,
                "X_train": X_train,
                "y_train": y_train,
                "groups_train": groups_train,
                "X_test": X_test,
                "y_test": y_test,
                "df_train_ref": df_train,
                "df_test_ref": df_test,
                "extra_params": {},
            })

            logger.debug("Configuración añadida: LightGBM")

        if models_cfg.get("lightgbm_without_hospadmtime", True):
            if "HospAdmTime" in X_train.columns:
                feature_cols_no_hosp = [c for c in feature_cols if c != "HospAdmTime"]
                X_search_no_hosp = search_data["X_search"][feature_cols_no_hosp].copy()

                configs.append({
                    "name": "LightGBM sin HospAdmTime",
                    "search_estimator": Pipeline([
                        ("clf", lgb.LGBMClassifier(
                            is_unbalance=True,
                            random_state=random_state,
                            n_jobs=estimator_n_jobs,
                            verbose=-1,
                        )),
                    ]),
                    "param_distributions": lgbm_param_dist,
                    "search_X": X_search_no_hosp,
                    "search_y": search_data["y_search"],
                    "search_groups": search_data["groups_search"],
                    "search_n_jobs": 1,
                    "final_estimator": make_lgbm_final,
                    "X_train": X_train[feature_cols_no_hosp].reset_index(drop=True),
                    "y_train": y_train,
                    "groups_train": groups_train,
                    "X_test": X_test[feature_cols_no_hosp].reset_index(drop=True),
                    "y_test": y_test,
                    "df_train_ref": df_train,
                    "df_test_ref": df_test,
                    "extra_params": {"excluded_features": ["HospAdmTime"]},
                })

                logger.debug("Configuración añadida: LightGBM sin HospAdmTime")

            else:
                logger.warning("HospAdmTime no está en las columnas. Se omite la ablación.")

    logger.debug("Total de configuraciones construidas: %d", len(configs))

    return configs

def main(params: dict) -> None:
    """Ejecuta el pipeline completo de entrenamiento y evaluación de modelos.

    Args:
        params (dict): Diccionario de configuración cargado desde params.yaml.
    """   

    random_state = int(params.get("random_state", 42))

    data_cfg = params.get("data", {})
    target_cfg = params.get("target", {})
    training_cfg = params.get("training", {})
    models_cfg = params.get("models", {})
    outputs_cfg = params.get("outputs", {})
    mlflow_cfg = params.get("mlflow", {})

    logger.debug("Parámetros de entrenamiento: %s", training_cfg)

    target_col = target_cfg.get("name", "SepsisLabel")
    patient_col = target_cfg.get("patient_col", "PatientID")
    aux_cols = target_cfg.get("aux_cols", ["PatientID", "Hospital", "TimeStep"])

    train_path = Path(data_cfg.get("train_path", "data/processed/train_preprocessed.parquet"))
    test_path = Path(data_cfg.get("test_path", "data/processed/test_preprocessed.parquet"))
    train_scaled_path = Path(
        data_cfg.get("train_scaled_path", "data/processed/train_preprocessed_scaled.parquet")
    )
    test_scaled_path = Path(
        data_cfg.get("test_scaled_path", "data/processed/test_preprocessed_scaled.parquet")
    )

    output_dirs = {
    "models_dir": Path(outputs_cfg.get("models_dir", "models")),
    "metrics_dir": Path(outputs_cfg.get("train_metrics_dir", "reports/metrics/train")),
    "predictions_dir": Path(outputs_cfg.get("predictions_dir", "reports/predictions")),
    }

    ensure_dirs(output_dirs.values())

    n_splits = int(training_cfg.get("n_splits", 5))
    n_iter_search = int(training_cfg.get("n_iter_search", 10))
    search_scoring = training_cfg.get("search_scoring", "average_precision")
    threshold_metric = metric_name(training_cfg.get("threshold_metric", "F1"))
    neg_pos_ratio = int(training_cfg.get("search_neg_pos_ratio", 4))
    search_n_jobs = int(training_cfg.get("search_n_jobs", -1))
    model_n_jobs = int(training_cfg.get("model_n_jobs", 2))
    search_verbose = int(training_cfg.get("search_verbose", 2))
    thresholds_grid = build_threshold_grid(training_cfg)

    # Si el search ya paraleliza, el estimador interno debe ir con n_jobs=1
    estimator_n_jobs = 1 if search_n_jobs != 1 else model_n_jobs

    mlflow_enabled = bool(mlflow_cfg.get("enabled", False))

    if mlflow_enabled:
        if mlflow is None:
            raise ImportError("MLflow está activado, pero no está instalado.")

        tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = mlflow_cfg.get("experiment_name", "sepsis-mlops")
        mlflow.set_experiment(experiment_name)
    else:
        logger.debug("MLflow desactivado. No se registrarán runs.")

    logger.info("Cargando datos...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    df_train_scaled = pd.read_parquet(train_scaled_path)
    df_test_scaled = pd.read_parquet(test_scaled_path)

    feature_cols = get_feature_columns(df_train, target_col, aux_cols)
    feature_cols_scaled = get_feature_columns(df_train_scaled, target_col, aux_cols)

    X_train = df_train[feature_cols].reset_index(drop=True)
    y_train = df_train[target_col].reset_index(drop=True)
    groups_train = df_train[patient_col].reset_index(drop=True)

    X_test = df_test[feature_cols].reset_index(drop=True)
    y_test = df_test[target_col].reset_index(drop=True)

    X_train_lr = df_train_scaled[feature_cols_scaled].reset_index(drop=True)
    y_train_lr = df_train_scaled[target_col].reset_index(drop=True)
    groups_train_lr = df_train_scaled[patient_col].reset_index(drop=True)

    X_test_lr = df_test_scaled[feature_cols_scaled].reset_index(drop=True)
    y_test_lr = df_test_scaled[target_col].reset_index(drop=True)

    logger.debug(
        "Datos cargados — train: %s filas | test: %s filas",
        len(df_train),
        len(df_test),
    )

    logger.info("Train sin escalar: %s", X_train.shape)
    logger.info("Test sin escalar:  %s", X_test.shape)
    logger.info("Train escalado:    %s", X_train_lr.shape)
    logger.info("Test escalado:     %s", X_test_lr.shape)

    search_data = prepare_search_subset(
        df_train=df_train,
        df_train_scaled=df_train_scaled,
        feature_cols=feature_cols,
        feature_cols_scaled=feature_cols_scaled,
        target_col=target_col,
        patient_col=patient_col,
        neg_pos_ratio=neg_pos_ratio,
        random_state=random_state,
    )

    logger.info("Submuestra búsqueda sin escalar: %s", search_data["X_search"].shape)
    logger.info("Submuestra búsqueda escalada:    %s", search_data["X_search_lr"].shape)

    all_results_row = {}
    all_results_patient = {}
    thresholds = {}
    training_times = {}
    
    with (mlflow.start_run(run_name="sepsis_training") if mlflow_enabled else contextlib.nullcontext()):
        # ============================================================
        # Baseline: DummyClassifier
        # ============================================================
        if models_cfg.get("dummy", True):
            with (mlflow.start_run(run_name="Baseline Dummy", nested=True) if mlflow_enabled else contextlib.nullcontext()):
                result = fit_evaluate_and_save(
                    model=DummyClassifier(strategy="stratified", random_state=random_state),
                    model_name="Baseline Dummy",
                    X_train=X_train,
                    y_train=y_train,
                    groups_train=groups_train,
                    X_test=X_test,
                    y_test=y_test,
                    df_train_ref=df_train,
                    df_test_ref=df_test,
                    threshold_metric=threshold_metric,
                    thresholds_grid=thresholds_grid,
                    n_splits=n_splits,
                    output_dirs=output_dirs,
                    target_col=target_col,
                    patient_col=patient_col,
                    model_params={"strategy": "stratified"},
                    tune_threshold=False,
                    fixed_threshold=0.5,
                    mlflow_enabled=mlflow_enabled,
                )

            all_results_row[result["model_name"]]     = result["metrics_row"]
            all_results_patient[result["model_name"]] = result["metrics_patient"]
            thresholds[result["model_name"]]          = result["threshold"]
            training_times[f"{result['model_name']} Retrain"] = result["retrain_time_seconds"]
            

        # ============================================================
        # Modelos con búsqueda de hiperparámetros
        # ============================================================

        model_configs = build_model_configs(
            models_cfg=models_cfg,
            random_state=random_state,
            estimator_n_jobs=estimator_n_jobs,
            model_n_jobs=model_n_jobs,
            search_n_jobs=search_n_jobs,
            search_data=search_data,
            X_train=X_train, y_train=y_train, groups_train=groups_train,
            X_test=X_test, y_test=y_test,
            df_train=df_train, df_test=df_test,
            X_train_lr=X_train_lr, y_train_lr=y_train_lr, groups_train_lr=groups_train_lr,
            X_test_lr=X_test_lr, y_test_lr=y_test_lr,
            df_train_scaled=df_train_scaled, df_test_scaled=df_test_scaled,
            feature_cols=feature_cols,
        )

        if not model_configs:
            logger.warning("No hay modelos con búsqueda de hiperparámetros activados en params.yaml.")

        for cfg in model_configs:
            with (mlflow.start_run(run_name=cfg["name"], nested=True) if mlflow_enabled else contextlib.nullcontext()):
                best_params, best_score, search_time = run_randomized_search(
                    model_name=cfg["name"],
                    estimator=cfg["search_estimator"],
                    param_distributions=cfg["param_distributions"],
                    X_search=cfg["search_X"],
                    y_search=cfg["search_y"],
                    groups_search=cfg["search_groups"],
                    n_iter=n_iter_search,
                    n_splits=n_splits,
                    scoring=search_scoring,
                    random_state=random_state,
                    n_jobs=cfg["search_n_jobs"],
                    verbose=search_verbose,
                    output_dirs=output_dirs,
                )

                training_times[f"{cfg['name']} Search"] = search_time

                result = fit_evaluate_and_save(
                    model=cfg["final_estimator"](best_params),
                    model_name=cfg["name"],
                    X_train=cfg["X_train"],
                    y_train=cfg["y_train"],
                    groups_train=cfg["groups_train"],
                    X_test=cfg["X_test"],
                    y_test=cfg["y_test"],
                    df_train_ref=cfg["df_train_ref"],
                    df_test_ref=cfg["df_test_ref"],
                    threshold_metric=threshold_metric,
                    thresholds_grid=thresholds_grid,
                    n_splits=n_splits,
                    output_dirs=output_dirs,
                    target_col=target_col,
                    patient_col=patient_col,
                    model_params={**best_params, "best_cv_score": best_score, **cfg["extra_params"]},
                    tune_threshold=True,
                    mlflow_enabled=mlflow_enabled,
                )

            all_results_row[result["model_name"]]     = result["metrics_row"]
            all_results_patient[result["model_name"]] = result["metrics_patient"]
            thresholds[result["model_name"]]          = result["threshold"]
            training_times[f"{result['model_name']} Retrain"] = result["retrain_time_seconds"]
        

        logger.debug("Modelos entrenados: %s", list(all_results_row.keys()))
        

    

    # ============================================================
    # Guardado de resúmenes globales
    # ============================================================
    if not all_results_row:
        raise RuntimeError("No se ha entrenado ningún modelo. Revisa params.yaml.")

    df_results_row = (
        pd.DataFrame(all_results_row)
        .T
        .sort_values("AUPRC", ascending=False)
    )

    df_results_patient = (
        pd.DataFrame(all_results_patient)
        .T
        .sort_values("AUPRC", ascending=False)
    )

    df_results_row.to_csv(Path(output_dirs["metrics_dir"]) / "all_model_metrics_row.csv")
    df_results_patient.to_csv(Path(output_dirs["metrics_dir"]) / "all_model_metrics_patient.csv")

    threshold_summary = (
        pd.DataFrame.from_dict(thresholds, orient="index", columns=["threshold"])
        .reset_index()
        .rename(columns={"index": "model_name"})
    )
    threshold_summary.to_csv(
        Path(output_dirs["metrics_dir"]) / "threshold_summary.csv",
        index=False,
    )

    training_times_df = (
        pd.DataFrame.from_dict(training_times, orient="index", columns=["seconds"])
        .reset_index()
        .rename(columns={"index": "stage"})
    )
    training_times_df.to_csv(
        Path(output_dirs["metrics_dir"]) / "training_times.csv",
        index=False,
    )

    best_model_name = df_results_row.index[0]

    experiment_config = {
        "best_model_by_row_auprc": best_model_name,
        "best_threshold": float(thresholds[best_model_name]),
        "random_state": random_state,
        "n_splits": n_splits,
        "n_iter_search": n_iter_search,
        "search_scoring": search_scoring,
        "threshold_metric": threshold_metric,
        "target_col": target_col,
        "patient_col": patient_col,
        "aux_cols": aux_cols,
        "feature_cols": feature_cols,
        "feature_cols_scaled": feature_cols_scaled,
        "metrics_row_test": all_results_row[best_model_name],
        "metrics_patient_test": all_results_patient[best_model_name],
    }

    experiment_config_path = Path(output_dirs["metrics_dir"]) / "experiment_config.json"

    save_json(
        experiment_config,
        experiment_config_path
    )

    if mlflow_enabled:
        mlflow.log_artifact(str(experiment_config_path))
        mlflow.log_metric("best_row_auprc", float(all_results_row[best_model_name].get("AUPRC", 0)))
        mlflow.log_param("best_model", best_model_name)

    logger.info("=" * 80)
    logger.info("Entrenamiento finalizado correctamente.")
    logger.info("=" * 80)
    logger.info("Mejor modelo por AUPRC a nivel fila: %s", best_model_name)
    logger.info("Umbral asociado: %.3f", thresholds[best_model_name])
    logger.info("Métricas guardadas en: %s", output_dirs["metrics_dir"])
    logger.info("Modelos guardados en: %s", output_dirs["models_dir"])
    logger.info("Predicciones guardadas en: %s", output_dirs["predictions_dir"])


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,  # nivel provisional hasta leer params
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Entrena modelos candidatos para detección de sepsis.")
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

    main(params)