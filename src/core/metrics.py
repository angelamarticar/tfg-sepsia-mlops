import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score


def _safe_round(value, ndigits: int = 4):
    """Redondea valores numéricos manteniendo np.nan si procede."""
    if pd.isna(value):
        return np.nan
    return round(float(value), ndigits)


def compute_metrics_patient(
    df_ref: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    target_col: str = "SepsisLabel",
    patient_col: str = "PatientID",
) -> dict:
    """Agrega predicciones por paciente y calcula métricas a ese nivel.

    Un paciente se considera positivo si tiene al menos una fila con target positivo.
    La probabilidad agregada por paciente se calcula como la probabilidad máxima
    predicha en cualquiera de sus registros horarios.

    Args:
        df_ref: DataFrame con columnas de paciente y variable objetivo.
        y_prob: Probabilidades predichas alineadas con df_ref.
        threshold: Umbral de decisión.
        target_col: Nombre de la columna objetivo.
        patient_col: Nombre de la columna identificadora del paciente.

    Returns:
        Diccionario con métricas agregadas a nivel de paciente.
    """
    if patient_col not in df_ref.columns:
        raise ValueError(f"df_ref debe contener la columna '{patient_col}'.")

    if target_col not in df_ref.columns:
        raise ValueError(f"df_ref debe contener la columna '{target_col}'.")

    y_prob = np.asarray(y_prob)

    if len(df_ref) != len(y_prob):
        raise ValueError(
            f"df_ref e y_prob deben tener la misma longitud. "
            f"Recibido: len(df_ref)={len(df_ref)}, len(y_prob)={len(y_prob)}."
        )

    df_eval = df_ref[[patient_col, target_col]].copy()
    df_eval["y_prob"] = y_prob

    df_patient = (
        df_eval
        .groupby(patient_col)
        .agg(
            y_true=(target_col, "max"),
            y_prob=("y_prob", "max"),
        )
        .reset_index()
    )

    y_true = df_patient["y_true"].to_numpy().astype(int)
    y_prob_agg = df_patient["y_prob"].to_numpy()
    y_pred = (y_prob_agg >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob_agg)
        auprc = average_precision_score(y_true, y_prob_agg)
    else:
        auroc = np.nan
        auprc = np.nan

    return {
        "Accuracy": _safe_round(accuracy),
        "Precision": _safe_round(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": _safe_round(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": _safe_round(specificity),
        "F1": _safe_round(f1_score(y_true, y_pred, zero_division=0)),
        "AUROC": _safe_round(auroc),
        "AUPRC": _safe_round(auprc),
        "n_patients": int(len(df_patient)),
        "n_positivos": int(y_true.sum()),
    }


def compute_metrics_row(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Calcula métricas a nivel de fila/registro horario.

    Args:
        y_true: Etiquetas reales a nivel fila.
        y_prob: Probabilidades predichas alineadas con y_true.
        threshold: Umbral de decisión para convertir probabilidades en clase binaria.

    Returns:
        Diccionario con métricas a nivel de registro horario.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true e y_prob deben tener la misma longitud. "
            f"Recibido: len(y_true)={len(y_true)}, len(y_prob)={len(y_prob)}."
        )

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan

    return {
        "Accuracy": _safe_round(accuracy),
        "Precision": _safe_round(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": _safe_round(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": _safe_round(specificity),
        "F1": _safe_round(f1_score(y_true, y_pred, zero_division=0)),
        "AUROC": _safe_round(auroc),
        "AUPRC": _safe_round(auprc),
        "n_rows": int(len(y_true)),
        "n_positivos": int(y_true.sum()),
    }


def cv_score(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scoring: str = "average_precision",
    n_splits: int = 5,
    n_jobs: int = -1,
) -> float:
    """Evalúa un pipeline con GroupKFold y devuelve la media del scoring indicado.

    Args:
        pipeline: Pipeline o estimador compatible con scikit-learn.
        X: Variables predictoras.
        y: Variable objetivo.
        groups: Identificador de grupo, normalmente PatientID.
        scoring: Métrica de evaluación de scikit-learn.
        n_splits: Número de particiones de GroupKFold.
        n_jobs: Número de trabajos en paralelo.

    Returns:
        Media de las puntuaciones de validación cruzada.
    """
    gkf = GroupKFold(n_splits=n_splits)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=gkf,
        groups=groups,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    return float(scores.mean())


def find_best_threshold_row(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "F1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Busca el umbral que maximiza una métrica a nivel de fila.

    Args:
        y_true: Etiquetas reales.
        y_prob: Probabilidades predichas.
        metric: Métrica a maximizar. Debe coincidir con una clave de compute_metrics_row.
        thresholds: Lista o array de umbrales candidatos.

    Returns:
        Mejor umbral y DataFrame con las métricas para cada umbral.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    results = []

    for threshold in thresholds:
        metrics = compute_metrics_row(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )

        results.append({
            "threshold": float(threshold),
            **metrics,
        })

    df_thresholds = pd.DataFrame(results)

    if metric not in df_thresholds.columns:
        available_metrics = list(df_thresholds.columns)
        raise ValueError(
            f"La métrica '{metric}' no existe. "
            f"Métricas disponibles: {available_metrics}"
        )

    best_idx = df_thresholds[metric].idxmax()
    best_threshold = float(df_thresholds.loc[best_idx, "threshold"])

    return best_threshold, df_thresholds