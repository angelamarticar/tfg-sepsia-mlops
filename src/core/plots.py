# src/core/plots.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)


def _ensure_parent_dir(path: str | Path) -> Path:
    """Crea la carpeta padre del archivo si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_roc_pr_curves(
    results: dict,
    output_path: str | Path,
) -> None:
    """Genera curvas ROC y Precision-Recall para varios modelos.

    Args:
        results: Diccionario con la forma:
            {
                "Modelo 1": {"y_true": array, "y_prob": array},
                "Modelo 2": {"y_true": array, "y_prob": array},
            }
        output_path: Ruta donde guardar la figura.
    """
    output_path = _ensure_parent_dir(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_roc, ax_pr = axes

    for model_name, values in results.items():
        y_true = np.asarray(values["y_true"]).astype(int)
        y_prob = np.asarray(values["y_prob"])

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax_roc.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUROC={roc_auc:.4f})",
        )

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        ax_pr.plot(
            recall,
            precision,
            label=f"{model_name} (AUPRC={auprc:.4f})",
        )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Aleatorio")
    ax_roc.set_title("Curva ROC")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)

    ax_pr.set_title("Curva Precision-Recall")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices_row_patient(
    y_true_row,
    y_prob_row,
    df_ref: pd.DataFrame,
    threshold: float,
    output_path: str | Path,
    target_col: str = "SepsisLabel",
    patient_col: str = "PatientID",
) -> None:
    """Genera matrices de confusión normalizadas a nivel fila y paciente.

    Args:
        y_true_row: Etiquetas reales a nivel de fila.
        y_prob_row: Probabilidades predichas a nivel de fila.
        df_ref: DataFrame alineado con y_prob_row, con PatientID y SepsisLabel.
        threshold: Umbral de decisión.
        output_path: Ruta donde guardar la figura.
        target_col: Nombre de la variable objetivo.
        patient_col: Nombre de la columna de paciente.
    """
    output_path = _ensure_parent_dir(output_path)

    y_true_row = np.asarray(y_true_row).astype(int)
    y_prob_row = np.asarray(y_prob_row)
    y_pred_row = (y_prob_row >= threshold).astype(int)

    cm_row = confusion_matrix(
        y_true_row,
        y_pred_row,
        labels=[0, 1],
        normalize="true",
    )

    df_eval = df_ref[[patient_col, target_col]].copy()
    df_eval["y_prob"] = y_prob_row

    df_patient = (
        df_eval
        .groupby(patient_col)
        .agg(
            y_true=(target_col, "max"),
            y_prob=("y_prob", "max"),
        )
        .reset_index()
    )

    y_true_patient = df_patient["y_true"].to_numpy().astype(int)
    y_pred_patient = (df_patient["y_prob"].to_numpy() >= threshold).astype(int)

    cm_patient = confusion_matrix(
        y_true_patient,
        y_pred_patient,
        labels=[0, 1],
        normalize="true",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp_row = ConfusionMatrixDisplay(
        confusion_matrix=cm_row,
        display_labels=["No sepsis", "Sepsis"],
    )
    disp_row.plot(ax=axes[0], values_format=".2f", colorbar=False)
    axes[0].set_title("Nivel registro horario")

    disp_patient = ConfusionMatrixDisplay(
        confusion_matrix=cm_patient,
        display_labels=["No sepsis", "Sepsis"],
    )
    disp_patient.plot(ax=axes[1], values_format=".2f", colorbar=False)
    axes[1].set_title("Nivel paciente")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    output_path: str | Path,
    top_n: int = 20,
    title: str = "Importancia de variables",
) -> None:
    """Genera un gráfico de importancia de variables para modelos tipo árbol.

    Args:
        model: Modelo entrenado o pipeline con un paso llamado 'clf'.
        feature_names: Lista de nombres de variables.
        output_path: Ruta donde guardar la figura.
        top_n: Número de variables más importantes a mostrar.
        title: Título de la figura.
    """
    output_path = _ensure_parent_dir(output_path)

    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
    else:
        clf = model

    if not hasattr(clf, "feature_importances_"):
        raise ValueError(
            "El modelo no tiene el atributo 'feature_importances_'. "
            "Esta función está pensada para modelos basados en árboles."
        )

    importances = np.asarray(clf.feature_importances_)

    if len(importances) != len(feature_names):
        raise ValueError(
            f"El número de importancias ({len(importances)}) no coincide "
            f"con el número de variables ({len(feature_names)})."
        )

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    df_importance = (
        df_importance
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(df_importance["feature"], df_importance["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    output_path_png: str | Path,
    output_path_pdf: str | Path | None = None,
    feature_names: list[str] | None = None,
    sample_size: int = 2000,
    random_state: int = 42,
    max_display: int = 20,
    title: str | None = None,
) -> None:
    """Genera un SHAP summary plot para modelos basados en árboles.

    Args:
        model: Modelo entrenado o pipeline con un paso llamado 'clf'.
        X: DataFrame con las variables usadas por el modelo.
        output_path_png: Ruta donde guardar la figura en PNG.
        output_path_pdf: Ruta opcional donde guardar la figura en PDF.
        feature_names: Nombres de variables. Si es None, usa X.columns.
        sample_size: Tamaño máximo de muestra para calcular SHAP.
        random_state: Semilla para muestreo.
        max_display: Número máximo de variables a mostrar.
        title: Título opcional de la figura.
    """
    import shap

    output_path_png = _ensure_parent_dir(output_path_png)

    if output_path_pdf is not None:
        output_path_pdf = _ensure_parent_dir(output_path_pdf)

    if feature_names is None:
        feature_names = list(X.columns)

    # Si el modelo es un pipeline, aplicamos los pasos previos al clasificador
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        preprocessing_steps = model.steps[:-1]

        X_tmp = X.copy()

        for _, step_obj in preprocessing_steps:
            X_tmp = step_obj.transform(X_tmp)

        X_shap = pd.DataFrame(X_tmp, columns=feature_names)
    else:
        clf = model
        X_shap = X.copy()

    sample_size = min(sample_size, len(X_shap))

    X_shap_sample = (
        X_shap
        .sample(n=sample_size, random_state=random_state)
        .astype(float)
    )

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_shap_sample)

    # Compatibilidad entre versiones de SHAP y modelos binarios
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    elif hasattr(shap_values, "values"):
        shap_values_plot = shap_values.values

        if shap_values_plot.ndim == 3:
            shap_values_plot = shap_values_plot[:, :, 1]
    else:
        shap_values_plot = shap_values

    if len(shap_values_plot.shape) == 3:
        shap_values_plot = shap_values_plot[:, :, 1]

    plt.figure(figsize=(8, 6))

    shap.summary_plot(
        shap_values_plot,
        X_shap_sample,
        show=False,
        max_display=max_display,
    )

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path_png, bbox_inches="tight", dpi=300)

    if output_path_pdf is not None:
        plt.savefig(output_path_pdf, bbox_inches="tight")

    plt.close()