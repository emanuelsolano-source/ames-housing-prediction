"""Limpieza semántica de valores faltantes y eliminación de outliers.

Reglas:
    * Las columnas donde "NA" significa "no existe el atributo" se rellenan
      con la categoría ``None`` (o ``0`` para años/áreas numéricas).
    * Los datos faltantes reales se dejan como nulos para imputarlos después
      dentro del pipeline de Scikit-Learn, ajustando solo sobre train.
    * Los outliers operacionales se eliminan únicamente de train.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import PROJECT_ROOT, RAW_DATA_DIR, TARGET_COL, read_raw_csv

PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

# "NA" = no existe el atributo (según data_description.txt).
# Columnas de un solo atributo: NA siempre significa "no existe".
SINGLE_NONE_COLS: tuple[str, ...] = (
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
)
# Grupos de columnas: "NA" en todo el grupo significa "no existe el elemento".
# Un NA parcial dentro del grupo es un dato faltante real y NO se rellena aquí.
GARAGE_GROUP_COLS: tuple[str, ...] = (
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
)
BASEMENT_GROUP_COLS: tuple[str, ...] = (
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
)
# Columnas numéricas que solo tienen sentido si existe garaje: se rellenan con 0.
ZERO_WHEN_ABSENT_COLS: tuple[str, ...] = ("GarageYrBlt",)

# Outliers operacionales: área muy grande con precio anormalmente bajo.
OUTLIER_AREA_COL: str = "GrLivArea"
OUTLIER_AREA_MIN: float = 4000.0
OUTLIER_PRICE_MAX: float = 300000.0


def fill_semantic_na(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena con ``None`` o ``0`` los nulos que representan ausencia de atributo.

    La columna ``GarageYrBlt`` solo se rellena con 0 cuando ``GarageType`` es
    ``None``, para no inventar un año de construcción de un garaje que no existe.

    Args:
        df: DataFrame crudo leído con ``read_raw_csv``.

    Returns:
        Copia del DataFrame con los nulos semánticos rellenados.
    """
    out: pd.DataFrame = df.copy()
    for col in SINGLE_NONE_COLS:
        out[col] = out[col].fillna("None")

    no_garage: pd.Series = out[GARAGE_GROUP_COLS[0]].isna()
    for col in GARAGE_GROUP_COLS:
        out.loc[no_garage, col] = "None"
    for col in ZERO_WHEN_ABSENT_COLS:
        out.loc[no_garage & out[col].isna(), col] = 0

    no_basement: pd.Series = out[list(BASEMENT_GROUP_COLS)].isna().all(axis=1)
    for col in BASEMENT_GROUP_COLS:
        out.loc[no_basement, col] = "None"
    return out


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina casas con área habitable muy grande y precio anormalmente bajo.

    Este filtro debe aplicarse solo a train, porque eliminar filas del test
    impediría predecir todas las propiedades que se evalúan.

    Args:
        df: DataFrame de entrenamiento que contiene ``GrLivArea`` y ``SalePrice``.

    Returns:
        DataFrame sin los outliers operacionales.
    """
    is_outlier: pd.Series = (df[OUTLIER_AREA_COL] > OUTLIER_AREA_MIN) & (
        df[TARGET_COL] < OUTLIER_PRICE_MAX
    )
    return df.loc[~is_outlier].reset_index(drop=True)


def build_clean_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datos crudos y aplica la limpieza a train y test.

    Returns:
        Tupla ``(train_clean, test_clean)``.
    """
    train: pd.DataFrame = read_raw_csv(RAW_DATA_DIR / "train.csv")
    test: pd.DataFrame = read_raw_csv(RAW_DATA_DIR / "test.csv")
    train_clean: pd.DataFrame = remove_outliers(fill_semantic_na(train))
    test_clean: pd.DataFrame = fill_semantic_na(test)
    return train_clean, test_clean


def main() -> None:
    """Genera ``train_clean.csv`` y ``test_clean.csv`` en ``data/processed``."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_clean, test_clean = build_clean_datasets()
    train_clean.to_csv(PROCESSED_DATA_DIR / "train_clean.csv", index=False)
    test_clean.to_csv(PROCESSED_DATA_DIR / "test_clean.csv", index=False)
    print(f"train_clean: {train_clean.shape[0]} filas x {train_clean.shape[1]} columnas")
    print(f"test_clean: {test_clean.shape[0]} filas x {test_clean.shape[1]} columnas")


if __name__ == "__main__":
    main()
