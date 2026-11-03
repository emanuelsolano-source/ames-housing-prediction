"""Definición de modelos baseline y avanzados con sus espacios de búsqueda.

Los hiperparámetros se buscan con ``RandomizedSearchCV`` dentro de cada fold
externo (validación anidada). Los nombres de parámetros usan el prefijo
``model__`` porque el estimador final va dentro del paso ``model`` del Pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from evaluation import RANDOM_STATE
from preprocessing import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    """Especificación de un modelo: estimador, espacio de búsqueda e iteraciones.

    Attributes:
        name: Nombre legible del modelo.
        estimator: Pipeline sin ajustar (preprocesamiento + modelo).
        param_distributions: Espacio de búsqueda. Vacío significa sin tuning.
        n_iter: Número de combinaciones aleatorias a evaluar.
    """

    name: str
    estimator: BaseEstimator
    param_distributions: dict[str, Any] = field(default_factory=dict)
    n_iter: int = 0


def _make_pipeline(model: BaseEstimator) -> Pipeline:
    """Envuelve un modelo con el preprocesamiento en un Pipeline.

    Args:
        model: Estimador de Scikit-Learn o XGBoost.

    Returns:
        Pipeline con los pasos ``preprocess`` y ``model``.
    """
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_model_specs() -> list[ModelSpec]:
    """Crea las especificaciones de los cinco modelos del proyecto.

    Returns:
        Lista de ``ModelSpec`` en el orden de evaluación.
    """
    return [
        ModelSpec(
            name="LinearRegression",
            estimator=_make_pipeline(LinearRegression()),
        ),
        ModelSpec(
            name="Ridge",
            estimator=_make_pipeline(Ridge()),
            param_distributions={"model__alpha": loguniform(1e-2, 1e3)},
            n_iter=30,
        ),
        ModelSpec(
            name="Lasso",
            estimator=_make_pipeline(Lasso(max_iter=50_000, random_state=RANDOM_STATE)),
            param_distributions={"model__alpha": loguniform(1e-4, 1e-1)},
            n_iter=30,
        ),
        ModelSpec(
            name="RandomForest",
            estimator=_make_pipeline(
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
            ),
            param_distributions={
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.3, 0.5],
            },
            n_iter=20,
        ),
        ModelSpec(
            name="XGBoost",
            estimator=_make_pipeline(
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            param_distributions={
                "model__n_estimators": randint(300, 1200),
                "model__learning_rate": loguniform(0.01, 0.1),
                "model__max_depth": randint(3, 7),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_weight": randint(1, 6),
                "model__reg_lambda": loguniform(0.1, 10.0),
            },
            n_iter=30,
        ),
    ]
