"""Carga y diagnóstico inicial de los datasets del proyecto Ames Housing.

Este módulo no transforma los datos: solo los lee desde ``data/raw`` y
reporta dimensiones, tipos de datos y distribución de la variable objetivo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
PROCESSED_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "train_clean.csv"
PROCESSED_TEST_PATH: Path = PROCESSED_DATA_DIR / "test_clean.csv"
RAW_FILES: tuple[str, ...] = ("train.csv", "test.csv", "sample_submission.csv")
TARGET_COL: str = "SalePrice"
# "NA" = ausencia del atributo o dato faltante. Las celdas vacías (al guardar
# CSV procesados con NaN reales) también son faltantes. "None" NO es nulo.
MISSING_TOKENS: list[str] = ["NA", ""]


def load_csv(file_path: Path) -> pd.DataFrame:
    """Lee un archivo CSV y falla de forma explícita si no existe.

    Args:
        file_path: Ruta absoluta o relativa al archivo CSV.

    Returns:
        DataFrame con el contenido del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    return pd.read_csv(file_path)


def read_raw_csv(file_path: Path) -> pd.DataFrame:
    """Lee un CSV tratando solo el token "NA" como nulo.

    Pandas trata "None" como nulo por defecto, pero en Ames "None" es una
    categoría válida (p. ej. MasVnrType = sin chapa). Por eso se desactivan
    los nulos por defecto y solo "NA" se convierte en valor faltante.

    Args:
        file_path: Ruta al archivo CSV.

    Returns:
        DataFrame con "NA" convertido a nulo y "None" conservado como texto.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    return pd.read_csv(file_path, keep_default_na=False, na_values=MISSING_TOKENS)


def load_raw_data(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Carga los archivos CSV crudos del proyecto.

    Args:
        data_dir: Carpeta que contiene los archivos crudos.

    Returns:
        Diccionario que mapea el nombre de archivo a su DataFrame.
    """
    return {name: load_csv(data_dir / name) for name in RAW_FILES}


def describe_dataset(df: pd.DataFrame, name: str) -> None:
    """Imprime dimensiones, conteo de tipos de datos y columnas con nulos.

    Args:
        df: DataFrame a diagnosticar.
        name: Nombre legible del dataset para el encabezado.
    """
    print(f"\n=== {name} ===")
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("Tipos de datos:")
    print(df.dtypes.value_counts().to_string())
    n_cols_with_na: int = int(df.isna().any().sum())
    print(f"Columnas con al menos un nulo: {n_cols_with_na}")


def describe_target(df: pd.DataFrame, target: str = TARGET_COL) -> None:
    """Imprime estadísticos descriptivos y el sesgo (skewness) de la variable objetivo.

    Args:
        df: DataFrame que contiene la variable objetivo.
        target: Nombre de la columna objetivo.

    Raises:
        KeyError: Si la columna objetivo no existe en el DataFrame.
    """
    if target not in df.columns:
        raise KeyError(f"La columna objetivo '{target}' no existe en el DataFrame.")
    y: pd.Series = df[target]
    print(f"\n=== Distribución de {target} ===")
    print(y.describe().to_string())
    print(f"Skewness (asimetría muestral): {y.skew():.4f}")
    print(f"Skewness de log(1 + y): {np.log1p(y).skew():.4f}")


def main() -> None:
    """Punto de entrada: carga los datos crudos y muestra el diagnóstico."""
    datasets: dict[str, pd.DataFrame] = load_raw_data()
    for file_name, df in datasets.items():
        describe_dataset(df, file_name)
    describe_target(datasets["train.csv"])


if __name__ == "__main__":
    main()
