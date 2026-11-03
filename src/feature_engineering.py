"""Ingeniería de características: codificación ordinal y variables de negocio.

Las transformaciones de este módulo son deterministas por fila: no aprenden
parámetros de los datos, por lo que no producen fuga de datos. La imputación
y la codificación One-Hot se ajustan después dentro del Pipeline de Scikit-Learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import PROCESSED_TRAIN_PATH, TARGET_COL, read_raw_csv

ID_COL: str = "Id"

# Escalas ordinales. "None" = atributo ausente, se codifica como 0.
QUALITY_MAP: dict[str, int] = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
EXPOSURE_MAP: dict[str, int] = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
BSMT_FIN_MAP: dict[str, int] = {
    "GLQ": 6,
    "ALQ": 5,
    "BLQ": 4,
    "Rec": 3,
    "LwQ": 2,
    "Unf": 1,
    "None": 0,
}
GARAGE_FINISH_MAP: dict[str, int] = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
CENTRAL_AIR_MAP: dict[str, int] = {"N": 0, "Y": 1}

ORDINAL_MAPPINGS: dict[str, dict[str, int]] = {
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "PoolQC": QUALITY_MAP,
    "BsmtExposure": EXPOSURE_MAP,
    "BsmtFinType1": BSMT_FIN_MAP,
    "BsmtFinType2": BSMT_FIN_MAP,
    "GarageFinish": GARAGE_FINISH_MAP,
    "CentralAir": CENTRAL_AIR_MAP,
}


def map_ordinal(series: pd.Series, mapping: dict[str, int], column: str) -> pd.Series:
    """Convierte una columna categórica ordinal en valores numéricos.

    Los nulos reales se conservan como NaN para imputarlos después. Si aparece
    una categoría no contemplada en el diccionario, se lanza un error en lugar
    de convertirla silenciosamente en NaN.

    Args:
        series: Columna categórica a mapear.
        mapping: Diccionario categoría -> valor numérico.
        column: Nombre de la columna, usado en el mensaje de error.

    Returns:
        Serie de tipo float64 con los valores mapeados.

    Raises:
        ValueError: Si hay categorías que no están en ``mapping``.
    """
    observed: set[str] = set(series.dropna().unique())
    unknown: set[str] = observed - set(mapping)
    if unknown:
        raise ValueError(f"Valores no mapeados en {column}: {sorted(unknown)}")
    return series.map(mapping).astype("float64")


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade variables de negocio derivadas de superficie, antigüedad y baños.

    Args:
        df: DataFrame con las columnas originales de Ames Housing.

    Returns:
        Copia del DataFrame con ``TotalSF``, ``Age``, ``IsRemodeled`` y ``TotalBaths``.
    """
    out: pd.DataFrame = df.copy()
    out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    out["Age"] = out["YrSold"] - out["YearBuilt"]
    out["IsRemodeled"] = (out["YearRemodAdd"] != out["YearBuilt"]).astype("int64")
    out["TotalBaths"] = (
        out["FullBath"]
        + 0.5 * out["HalfBath"]
        + out["BsmtFullBath"]
        + 0.5 * out["BsmtHalfBath"]
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica codificación ordinal y variables de negocio.

    Args:
        df: DataFrame limpio (salida de ``cleaning.py``) de train o test.

    Returns:
        DataFrame con las columnas ordinales numéricas y las nuevas variables.
    """
    out: pd.DataFrame = df.copy()
    for column, mapping in ORDINAL_MAPPINGS.items():
        out[column] = map_ordinal(out[column], mapping, column)
    return add_business_features(out)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa predictores y objetivo en escala logarítmica.

    Se usa log(1 + SalePrice) como objetivo, según la base matemática de la Fase 1.
    ``Id`` no es un predictor y se excluye.

    Args:
        df: DataFrame de entrenamiento con ``SalePrice``.

    Returns:
        Tupla ``(X, y_log)``.

    Raises:
        KeyError: Si falta la columna objetivo.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Falta la columna objetivo '{TARGET_COL}'.")
    X: pd.DataFrame = df.drop(columns=[ID_COL, TARGET_COL])
    y_log: pd.Series = pd.Series(np.log1p(df[TARGET_COL]), index=df.index, name="log_SalePrice")
    return X, y_log


def main() -> None:
    """Muestra la correlación de las nuevas variables con el objetivo logarítmico."""
    train_clean: pd.DataFrame = read_raw_csv(PROCESSED_TRAIN_PATH)
    features: pd.DataFrame = build_features(train_clean)
    _, y_log = split_features_target(features)
    new_cols: list[str] = ["TotalSF", "Age", "IsRemodeled", "TotalBaths", "OverallQual", "GrLivArea"]
    print("Rango de Age:", features["Age"].min(), "a", features["Age"].max())
    print("Nulos en nuevas variables:\n", features[new_cols].isna().sum().to_string())
    corr: pd.Series = features[new_cols].corrwith(y_log).sort_values(ascending=False)
    print("\nCorrelación de Pearson con log(1+SalePrice):")
    print(corr.round(4).to_string())


if __name__ == "__main__":
    main()
