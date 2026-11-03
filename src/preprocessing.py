"""Pipeline de preprocesamiento: imputación, escalado y codificación.

Todas las transformaciones que aprenden parámetros (medianas, modas, escalas
y categorías) viven dentro de un ``Pipeline``. Así, al usarlo en validación
cruzada, cada fold ajusta esos parámetros solo con sus datos de entrenamiento,
evitando la fuga de datos.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Variables guardadas como números pero que son categorías nominales (código de tipo de vivienda).
NOMINAL_NUMERIC_COLS: tuple[str, ...] = ("MSSubClass",)


def prepare_predictors(X: pd.DataFrame) -> pd.DataFrame:
    """Convierte códigos numéricos nominales a texto para que se traten como categorías.

    Args:
        X: DataFrame de predictores (sin ``Id`` ni objetivo).

    Returns:
        Copia del DataFrame con las columnas nominales convertidas a texto.
    """
    out: pd.DataFrame = X.copy()
    for col in NOMINAL_NUMERIC_COLS:
        if col in out.columns:
            out[col] = out[col].astype("str")
    return out


def build_preprocessor() -> ColumnTransformer:
    """Construye el transformador de columnas numéricas y categóricas.

    - Numéricas: imputación con la mediana (robusta a colas largas) y ``RobustScaler``
      (centra con la mediana y escala con el rango intercuartílico).
    - Categóricas: imputación con la moda y One-Hot Encoding. Las categorías no vistas
      en entrenamiento se codifican como ceros en lugar de provocar un error.

    Las columnas se seleccionan al momento del ajuste según su tipo, así que el
    transformador no depende de listas fijas ni de datos externos.

    Returns:
        ``ColumnTransformer`` sin ajustar.
    """
    numeric_pipeline: Pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    categorical_pipeline: Pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
