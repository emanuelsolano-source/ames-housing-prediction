"""Auditoría de valores faltantes en los datasets de Ames Housing.

Genera un resumen por columna con el número y porcentaje de nulos en
``train.csv`` y ``test.csv``. No modifica los datos.
"""

from __future__ import annotations

import pandas as pd

from data_loader import RAW_DATA_DIR, TARGET_COL, read_raw_csv


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el conteo y porcentaje de nulos por columna.

    Args:
        df: DataFrame a auditar.

    Returns:
        DataFrame con columnas ``n_null`` y ``pct_null``, solo para columnas
        con al menos un nulo, ordenado de mayor a menor porcentaje.
    """
    counts: pd.Series = df.isna().sum()
    summary: pd.DataFrame = pd.DataFrame(
        {"n_null": counts, "pct_null": (counts / len(df) * 100).round(2)}
    )
    summary = summary[summary["n_null"] > 0]
    return summary.sort_values("pct_null", ascending=False)


def main() -> None:
    """Imprime la auditoría de nulos de train y test."""
    train: pd.DataFrame = read_raw_csv(RAW_DATA_DIR / "train.csv")
    test: pd.DataFrame = read_raw_csv(RAW_DATA_DIR / "test.csv")
    print("=== Nulos en train.csv ===")
    print(null_summary(train.drop(columns=[TARGET_COL])).to_string())
    print("\n=== Nulos en test.csv ===")
    print(null_summary(test).to_string())


if __name__ == "__main__":
    main()
