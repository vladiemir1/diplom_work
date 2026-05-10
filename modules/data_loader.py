from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251")
OPTIONAL_COLUMNS = ("product_name", "rating", "date")


class DataLoadError(ValueError):
    """Raised when an input reviews file cannot be read or normalized."""


def _read_source_bytes(source: Any) -> tuple[str, bytes]:
    if hasattr(source, "getvalue"):
        name = getattr(source, "name", "uploaded_file")
        return name, source.getvalue()

    path = Path(source)
    return path.name, path.read_bytes()


def load_reviews(source: Any) -> pd.DataFrame:
    """Load reviews from a CSV or XLSX source.

    The source may be a Streamlit UploadedFile, a file-like object with
    getvalue(), or a local path.
    """

    file_name, data = _read_source_bytes(source)
    suffix = Path(file_name).suffix.lower()

    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in CSV_ENCODINGS:
            try:
                return pd.read_csv(BytesIO(data), encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
        raise DataLoadError(f"Не удалось прочитать CSV-файл. Последняя ошибка: {last_error}")

    if suffix == ".xlsx":
        try:
            return pd.read_excel(BytesIO(data))
        except Exception as exc:  # pragma: no cover - exact engine errors vary
            raise DataLoadError(f"Не удалось прочитать XLSX-файл: {exc}") from exc

    raise DataLoadError("Поддерживаются только файлы CSV и XLSX.")


def validate_reviews_df(df: pd.DataFrame, text_column: str | None = "review_text") -> tuple[bool, list[str]]:
    errors: list[str] = []
    if df.empty:
        errors.append("Файл не содержит строк.")

    if text_column and text_column not in df.columns:
        errors.append(f"В таблице нет выбранного текстового столбца: {text_column}.")

    return not errors, errors


def selectable_text_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable for review text selection."""

    candidates = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]
    return candidates or [str(column) for column in df.columns]


def ensure_review_id(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "review_id" not in result.columns:
        result.insert(0, "review_id", range(1, len(result) + 1))
    return result


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result


def prepare_reviews_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Normalize user data to the app contract while preserving extra columns."""

    if text_column not in df.columns:
        raise DataLoadError(f"Выбранный текстовый столбец не найден: {text_column}")

    result = df.copy()
    if text_column != "review_text":
        result["review_text"] = result[text_column]

    result = ensure_review_id(result)

    for column in OPTIONAL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    return normalize_date_column(result)

