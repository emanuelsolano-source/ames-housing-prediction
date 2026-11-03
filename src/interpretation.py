"""Diagnóstico e interpretación: residuos, correlaciones, importancia y SHAP.

Los residuos se calculan sobre predicciones fuera de muestra (OOF), no sobre el
ajuste en train, para que el diagnóstico refleje el error de generalización.
La importancia y SHAP se calculan sobre un único ajuste de XGBoost en todo el train.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from data_loader import PROJECT_ROOT
from models import ModelSpec, build_model_specs
from train import build_search, load_datasets

FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
TOP_N: int = 15


def plot_residuals(y_log: np.ndarray, oof_pred: np.ndarray, path: Path) -> None:
    """Dibuja residuos frente a predichos para detectar sesgos y heterocedasticidad.

    Args:
        y_log: Valores reales en escala logarítmica.
        oof_pred: Predicciones OOF en escala logarítmica.
        path: Ruta de salida de la figura.
    """
    residuals: np.ndarray = y_log - oof_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(oof_pred, residuals, s=12, alpha=0.5)
    ax.axhline(0.0, color="red", linewidth=1)
    ax.set_xlabel("Predicción log(1 + SalePrice)")
    ax.set_ylabel("Residuo (real - predicho)")
    ax.set_title("Residuos vs. predichos (OOF, ensamble)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(features: pd.DataFrame, path: Path, top_k: int = 20) -> None:
    """Dibuja el mapa de calor de las variables numéricas más correlacionadas con el objetivo.

    Args:
        features: DataFrame con predictores numéricos y ``log_SalePrice``.
        path: Ruta de salida de la figura.
        top_k: Número de variables a incluir, ordenadas por correlación absoluta.
    """
    corr: pd.DataFrame = features.corr(numeric_only=True)
    top_cols: list[str] = (
        corr["log_SalePrice"].abs().sort_values(ascending=False).head(top_k + 1).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr.loc[top_cols, top_cols], cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f"Correlaciones de Pearson: top {top_k} variables con log(1 + SalePrice)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fit_xgboost_final(X: pd.DataFrame, y_log: pd.Series, spec: ModelSpec) -> Pipeline:
    """Ajusta XGBoost sobre todo el train con los hiperparámetros de su búsqueda.

    Args:
        X: Predictores de train.
        y_log: Objetivo en escala logarítmica.
        spec: Especificación del modelo.

    Returns:
        Pipeline ajustado (preprocesamiento + XGBoost).

    Raises:
        TypeError: Si el ajuste no devuelve un Pipeline.
    """
    search: BaseEstimator = build_search(spec)
    search.fit(X, y_log)
    fitted: BaseEstimator = search.best_estimator_
    if not isinstance(fitted, Pipeline):
        raise TypeError("Se esperaba un Pipeline ajustado.")
    return fitted



def native_importance(pipeline: Pipeline) -> pd.Series:
    """Devuelve la importancia nativa de XGBoost por columna transformada.

    Args:
        pipeline: Pipeline ajustado con los pasos ``preprocess`` y ``model``.

    Returns:
        Serie de importancias ordenada de mayor a menor.
    """
    names: np.ndarray = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance: np.ndarray = pipeline.named_steps["model"].feature_importances_
    return pd.Series(importance, index=names).sort_values(ascending=False)


def shap_values(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Calcula valores SHAP exactos para árboles sobre el espacio transformado.

    Args:
        pipeline: Pipeline ajustado.
        X: Predictores de train.

    Returns:
        Tupla ``(shap_matrix, X_transformed)`` con columnas coherentes.
    """
    preprocess = pipeline.named_steps["preprocess"]
    names: np.ndarray = preprocess.get_feature_names_out()
    X_df: pd.DataFrame = pd.DataFrame(preprocess.transform(X), columns=names)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    return explainer.shap_values(X_df), X_df


def plot_top_importance(importance: pd.Series, path: Path, top_n: int = TOP_N) -> None:
    """Dibuja las variables más importantes con SHAP medio absoluto.

    Args:
        importance: Serie de importancia por variable.
        path: Ruta de salida de la figura.
        top_n: Número de variables a mostrar.
    """
    top: pd.Series = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values)
    ax.set_xlabel("SHAP medio absoluto (escala log(1 + SalePrice))")
    ax.set_title(f"Top {top_n} características clave")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_ensemble_oof(reports_dir: Path) -> np.ndarray:
    """Reconstruye las predicciones OOF del ensamble NNLS a partir de los reportes.

    Args:
        reports_dir: Carpeta ``reports`` generada por ``train.py``.

    Returns:
        Array con predicciones OOF del ensamble en escala logarítmica.
    """
    oof: pd.DataFrame = pd.read_csv(reports_dir / "oof_predictions.csv")
    weights: pd.DataFrame = pd.read_csv(reports_dir / "ensemble_weights.csv")
    ordered: np.ndarray = oof[weights["model"].tolist()].to_numpy()
    return ordered @ weights["weight"].to_numpy()


def main() -> None:
    """Genera residuos, correlaciones, importancia y SHAP en ``reports/figures``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    oof_pred: np.ndarray = load_ensemble_oof(REPORTS_DIR)
    y_oof: np.ndarray = pd.read_csv(REPORTS_DIR / "oof_predictions.csv")["y_log"].to_numpy()
    plot_residuals(y_oof, oof_pred, FIGURES_DIR / "residuals_vs_predicted.png")

    X, y_series, _, _ = load_datasets()
    corr_df: pd.DataFrame = X.select_dtypes(include="number").assign(log_SalePrice=y_series.to_numpy())
    plot_correlation_heatmap(corr_df, FIGURES_DIR / "correlation_heatmap.png")

    spec_xgb: ModelSpec = next(s for s in build_model_specs() if s.name == "XGBoost")
    pipeline: Pipeline = fit_xgboost_final(X, y_series, spec_xgb)
    native: pd.Series = native_importance(pipeline)
    native.to_csv(REPORTS_DIR / "feature_importance_xgboost.csv", header=["importance"])

    shap_matrix, X_trans = shap_values(pipeline, X)
    shap_mean: pd.Series = pd.Series(np.abs(shap_matrix).mean(axis=0), index=X_trans.columns)
    shap_mean = shap_mean.sort_values(ascending=False)
    shap_mean.to_csv(REPORTS_DIR / "shap_mean_abs.csv", header=["mean_abs_shap"])
    plot_top_importance(shap_mean, FIGURES_DIR / "top15_features.png")

    print("Top 15 por SHAP medio absoluto:")
    print(shap_mean.head(TOP_N).round(4).to_string())


if __name__ == "__main__":
    main()
