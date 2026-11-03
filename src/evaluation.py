"""Métricas y esquemas de validación cruzada.

La métrica principal es RMSE sobre log(1 + SalePrice). Como la variable objetivo
ya está en escala logarítmica, esta métrica penaliza los errores relativos
(porcentuales) de forma similar en casas baratas y caras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

RANDOM_STATE: int = 42
N_OUTER_SPLITS: int = 5
N_INNER_SPLITS: int = 3
N_TARGET_BINS: int = 10


def rmse_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula RMSE en escala logarítmica.

    Args:
        y_true: Valores reales de log(1 + SalePrice).
        y_pred: Predicciones en la misma escala.

    Returns:
        Raíz del error cuadrático medio.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_target_bins(y_log: pd.Series, n_bins: int = N_TARGET_BINS) -> np.ndarray:
    """Discretiza el objetivo en cuantiles para estratificar un problema de regresión.

    Estratificar por deciles asegura que cada fold tenga una distribución de
    precios parecida a la del conjunto completo.

    Args:
        y_log: Objetivo en escala logarítmica.
        n_bins: Número de cuantiles.

    Returns:
        Array de etiquetas enteras de bin.
    """
    return pd.qcut(y_log, q=n_bins, labels=False, duplicates="drop").to_numpy()


def make_outer_splits(X: pd.DataFrame, y_log: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera los folds externos K-Fold estratificados por deciles del objetivo.

    Args:
        X: Predictores.
        y_log: Objetivo en escala logarítmica.

    Returns:
        Lista de tuplas ``(indices_train, indices_test)``.
    """
    cv: StratifiedKFold = StratifiedKFold(
        n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    return list(cv.split(X, make_target_bins(y_log)))


def make_inner_cv() -> KFold:
    """Crea el esquema interno para la búsqueda de hiperparámetros.

    Returns:
        ``KFold`` con mezcla y semilla fija.
    """
    return KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def fold_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Calcula el RMSE de cada fold externo a partir de predicciones fuera de muestra.

    Args:
        y_true: Valores reales.
        y_pred: Predicciones fuera de muestra.
        splits: Folds externos usados para generar ``y_pred``.

    Returns:
        Array con un RMSE por fold.
    """
    return np.array([rmse_log(y_true[test], y_pred[test]) for _, test in splits])
