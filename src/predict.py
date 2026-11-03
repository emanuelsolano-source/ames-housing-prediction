"""Predicción final sobre test: reversión del logaritmo y validación de la entrega.

Las predicciones del ensamble están en escala log(1 + SalePrice). Se revierten
con exp(z) - 1 y se validan contra ``sample_submission.csv`` (mismos Id, mismo
orden y sin nulos ni valores no finitos).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import PROJECT_ROOT, RAW_DATA_DIR, read_raw_csv
from feature_engineering import ID_COL

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
SUBMISSION_PATH: Path = REPORTS_DIR / "submission.csv"
PRED_COL: str = "SalePrice"


def inverse_log_target(z: np.ndarray) -> np.ndarray:
    """Revierte la transformación log(1 + y) con exp(z) - 1.

    Args:
        z: Predicciones en escala logarítmica.

    Returns:
        Predicciones de precio en dólares.
    """
    return np.expm1(z)


def build_submission(log_pred: pd.DataFrame) -> pd.DataFrame:
    """Convierte predicciones logarítmicas en el formato de entrega de Kaggle.

    Args:
        log_pred: DataFrame con columnas ``Id`` y ``log_pred``.

    Returns:
        DataFrame con columnas ``Id`` y ``SalePrice``.
    """
    return pd.DataFrame(
        {ID_COL: log_pred[ID_COL], PRED_COL: inverse_log_target(log_pred["log_pred"].to_numpy())}
    )


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Verifica que la entrega coincida con la estructura de ``sample_submission.csv``.

    Args:
        submission: Predicciones a validar.
        sample: Archivo de ejemplo oficial.

    Raises:
        ValueError: Si las columnas, los Id, el orden o los valores no son válidos.
    """
    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"Columnas incorrectas: {list(submission.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"Filas: {len(submission)} != {len(sample)}")
    if not submission[ID_COL].equals(sample[ID_COL]):
        raise ValueError("Los Id no coinciden en orden con sample_submission.csv")
    values: np.ndarray = submission[PRED_COL].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Hay predicciones nulas, infinitas o no positivas.")


def main() -> None:
    """Genera ``reports/submission.csv`` y lo valida contra la plantilla oficial."""
    log_pred: pd.DataFrame = pd.read_csv(REPORTS_DIR / "test_predictions_log.csv")
    sample: pd.DataFrame = read_raw_csv(RAW_DATA_DIR / "sample_submission.csv")
    submission: pd.DataFrame = build_submission(log_pred)
    validate_submission(submission, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Entrega validada: {len(submission)} filas -> {SUBMISSION_PATH}")
    print(submission[PRED_COL].describe().round(0).to_string())


if __name__ == "__main__":
    main()
