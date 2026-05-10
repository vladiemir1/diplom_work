from __future__ import annotations

from io import BytesIO

import pandas as pd


RESULT_EXPORT_COLUMNS = [
    "review_id",
    "product_name",
    "rating",
    "date",
    "review_text",
    "processed_text",
    "aspect",
    "sentiment",
    "sentiment_label",
    "confidence",
    "explanation",
]


def _existing_columns(df: pd.DataFrame, desired: list[str]) -> list[str]:
    return [column for column in desired if column in df.columns]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    export_df = df[_existing_columns(df, RESULT_EXPORT_COLUMNS)] if not df.empty else df
    return export_df.to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(
    raw_df: pd.DataFrame | None = None,
    processed_df: pd.DataFrame | None = None,
    results_df: pd.DataFrame | None = None,
    agg_df: pd.DataFrame | None = None,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sheets = {
            "Исходные данные": raw_df,
            "Обработанные данные": processed_df,
            "Результаты анализа": results_df,
            "Агрегаты": agg_df,
        }
        wrote_any = False
        for name, df in sheets.items():
            if df is not None:
                df.to_excel(writer, sheet_name=name, index=False)
                wrote_any = True
        if not wrote_any:
            pd.DataFrame().to_excel(writer, sheet_name="Данные", index=False)
    return buffer.getvalue()

