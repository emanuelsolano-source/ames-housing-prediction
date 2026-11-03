"""Entrenamiento con validación anidada, combinación de modelos y predicción final.

Flujo:
    1. Validación anidada por modelo: en cada fold externo, ``RandomizedSearchCV``
       busca hiperparámetros con validación interna y evalúa en el fold externo.
       Así la estimación del error no depende de haber elegido hiperparámetros
       mirando esos mismos datos.
    2. Combinación: se pesan los modelos con predicciones fuera de muestra (OOF).
       Los pesos se evalúan con una validación cruzada sobre las predicciones OOF.
    3. Ajuste final: cada modelo se re-ajusta sobre todo el train y se predice test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RandomizedSearchCV

from data_loader import PROCESSED_TEST_PATH, PROCESSED_TRAIN_PATH, PROJECT_ROOT, read_raw_csv
from evaluation import (
    RANDOM_STATE,
    fold_rmse,
    make_inner_cv,
    make_outer_splits,
)
from feature_engineering import ID_COL, build_features, split_features_target
from models import ModelSpec, build_model_specs
from preprocessing import prepare_predictors

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
SCORING: str = "neg_root_mean_squared_error"


def load_datasets() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga y transforma train y test con la ingeniería de características.

    Returns:
        Tupla ``(X_train, y_log_train, X_test, test_ids)``.
    """
    train: pd.DataFrame = build_features(read_raw_csv(PROCESSED_TRAIN_PATH))
    test: pd.DataFrame = build_features(read_raw_csv(PROCESSED_TEST_PATH))
    X_train, y_log = split_features_target(train)
    X_train = prepare_predictors(X_train).reset_index(drop=True)
    y_log = y_log.reset_index(drop=True)
    X_test: pd.DataFrame = prepare_predictors(test.drop(columns=[ID_COL]))
    test_ids: pd.Series = test[ID_COL]
    return X_train, y_log, X_test, test_ids


def build_search(spec: ModelSpec) -> BaseEstimator:
    """Crea el estimador a ajustar: búsqueda aleatoria o modelo fijo.

    Args:
        spec: Especificación del modelo.

    Returns:
        ``RandomizedSearchCV`` si el modelo tiene espacio de búsqueda; si no, una copia sin ajustar.
    """
    if not spec.param_distributions:
        return clone(spec.estimator)
    return RandomizedSearchCV(
        estimator=clone(spec.estimator),
        param_distributions=spec.param_distributions,
        n_iter=spec.n_iter,
        cv=make_inner_cv(),
        scoring=SCORING,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )


def nested_cv(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta validación anidada para un modelo.

    Args:
        spec: Especificación del modelo.
        X: Predictores de train.
        y: Objetivo en escala logarítmica.
        splits: Folds externos.

    Returns:
        Tupla ``(oof_predictions, fold_rmse)``.
    """
    oof: np.ndarray = np.zeros(len(y))
    for train_idx, test_idx in splits:
        estimator: BaseEstimator = build_search(spec)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = estimator.predict(X.iloc[test_idx])
    return oof, fold_rmse(y.to_numpy(), oof, splits)


def fit_nnls_weights(oof_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estima pesos no negativos que minimizan el error cuadrático de la combinación.

    Args:
        oof_matrix: Matriz (n_muestras x n_modelos) de predicciones OOF.
        y: Valores reales en escala logarítmica.

    Returns:
        Vector de pesos no negativos, normalizado para sumar 1.
    """
    weights, _ = nnls(oof_matrix, y)
    total: float = float(weights.sum())
    if total <= 0:
        return np.full(oof_matrix.shape[1], 1 / oof_matrix.shape[1])
    return weights / total


def evaluate_blend(
    oof_matrix: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Evalúa la combinación con pesos aprendidos fuera de cada fold.

    Los pesos de cada fold se ajustan con las predicciones OOF de los otros folds
    y se miden en el fold restante, así el error no usa la misma muestra para
    ajustar y evaluar los pesos.

    Args:
        oof_matrix: Predicciones OOF (n_muestras x n_modelos).
        y: Valores reales en escala logarítmica.
        splits: Folds externos.

    Returns:
        Array con el RMSE de la combinación por fold.
    """
    blended: np.ndarray = np.zeros(len(y))
    for train_idx, test_idx in splits:
        weights: np.ndarray = fit_nnls_weights(oof_matrix[train_idx], y[train_idx])
        blended[test_idx] = oof_matrix[test_idx] @ weights
    return fold_rmse(y, blended, splits)


def _summary_row(name: str, folds: np.ndarray) -> dict[str, object]:
    """Resume el RMSE por fold de un modelo o combinación.

    Args:
        name: Nombre del modelo o combinación.
        folds: RMSE por fold externo.

    Returns:
        Diccionario con media y desviación estándar muestral.
    """
    return {"model": name, "rmse_log_mean": folds.mean(), "rmse_log_std": folds.std(ddof=1)}


def main() -> None:
    """Ejecuta validación anidada, combinación y predicción final."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y_log, X_test, test_ids = load_datasets()
    splits = make_outer_splits(X, y_log)
    y_arr: np.ndarray = y_log.to_numpy()

    specs: list[ModelSpec] = build_model_specs()
    oof_columns: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []
    for spec in specs:
        print(f"Validación anidada: {spec.name}", flush=True)
        oof, folds = nested_cv(spec, X, y_log, splits)
        oof_columns[spec.name] = oof
        summary_rows.append(_summary_row(spec.name, folds))

    oof_matrix: np.ndarray = np.column_stack([oof_columns[s.name] for s in specs])
    avg_folds: np.ndarray = fold_rmse(y_arr, oof_matrix.mean(axis=1), splits)
    summary_rows.append(_summary_row("Ensamble_promedio", avg_folds))
    summary_rows.append(_summary_row("Ensamble_NNLS", evaluate_blend(oof_matrix, y_arr, splits)))

    summary: pd.DataFrame = pd.DataFrame(summary_rows).sort_values("rmse_log_mean")
    summary.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    oof_df: pd.DataFrame = pd.DataFrame(oof_columns)
    oof_df["y_log"] = y_arr
    oof_df.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)
    print(summary.round(5).to_string(index=False))

    final_weights: np.ndarray = fit_nnls_weights(oof_matrix, y_arr)
    weights_df: pd.DataFrame = pd.DataFrame({"model": [s.name for s in specs], "weight": final_weights})
    weights_df.to_csv(REPORTS_DIR / "ensemble_weights.csv", index=False)
    print("\nPesos NNLS:")
    print(weights_df.round(4).to_string(index=False))

    test_preds: list[np.ndarray] = []
    for spec in specs:
        print(f"Ajuste final: {spec.name}", flush=True)
        model: BaseEstimator = build_search(spec)
        model.fit(X, y_log)
        test_preds.append(model.predict(X_test))
    pred_log: np.ndarray = np.column_stack(test_preds) @ final_weights
    pd.DataFrame({ID_COL: test_ids, "log_pred": pred_log}).to_csv(
        REPORTS_DIR / "test_predictions_log.csv", index=False
    )
    print("Predicciones guardadas en reports/test_predictions_log.csv")


if __name__ == "__main__":
    main()
