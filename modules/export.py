from __future__ import annotations

from io import BytesIO

import pandas as pd


ASPECT_COLUMN_MAP = {
    "Качество товара": "quality",
    "Соответствие описанию": "description_match",
    "Упаковка": "packaging",
    "Доставка": "delivery",
}

WIDE_EXPORT_COLUMNS = [
    "review_id",
    "product_name",
    "rating",
    "date",
    "review_text",
    "quality",
    "description_match",
    "packaging",
    "delivery",
    "overall_sentiment",
]


def _existing_columns(df: pd.DataFrame, desired: list[str]) -> list[str]:
    return [column for column in desired if column in df.columns]


def _calc_overall_sentiment(row: pd.Series) -> str:
    sentiments = set()
    for col in ASPECT_COLUMN_MAP.values():
        value = row.get(col, "none")
        if value and value != "none":
            sentiments.add(value)
    if not sentiments:
        return "undefined"
    if len(sentiments) == 1:
        return sentiments.pop()
    if "positive" in sentiments and "negative" in sentiments:
        return "mixed"
    if "negative" in sentiments:
        return "negative"
    if "positive" in sentiments:
        return "positive"
    return "neutral"


def to_wide_format(
    results_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert long-format aspect results to wide format (one row per review)."""

    base_cols = ["review_id", "product_name", "rating", "date", "review_text"]

    if results_df.empty:
        cols = base_cols + list(ASPECT_COLUMN_MAP.values()) + ["overall_sentiment"]
        return pd.DataFrame(columns=cols)

    # Exclude legacy "Общее впечатление" if present
    filtered = results_df[results_df["aspect"] != "Общее впечатление"].copy()

    existing_base = [c for c in base_cols if c in results_df.columns]
    base_df = results_df[existing_base].drop_duplicates(subset=["review_id"])

    # If valid_df is provided, include reviews that had no aspects detected
    if valid_df is not None and not valid_df.empty:
        valid_base = valid_df[_existing_columns(valid_df, base_cols)].copy()
        if "review_id" in valid_base.columns:
            valid_base["review_id"] = valid_base["review_id"].astype(str)
        if "review_id" in base_df.columns:
            base_df["review_id"] = base_df["review_id"].astype(str)
        # Merge: keep all valid reviews, even those without aspect results
        missing = valid_base[~valid_base["review_id"].isin(base_df["review_id"])]
        base_df = pd.concat([base_df, missing], ignore_index=True)

    wide = base_df.copy()

    for aspect_name, col_name in ASPECT_COLUMN_MAP.items():
        if not filtered.empty:
            aspect_data = (
                filtered[filtered["aspect"] == aspect_name][["review_id", "sentiment"]]
                .drop_duplicates(subset=["review_id"], keep="first")
                .rename(columns={"sentiment": col_name})
            )
            if "review_id" in aspect_data.columns:
                aspect_data["review_id"] = aspect_data["review_id"].astype(str)
            wide = wide.merge(aspect_data, on="review_id", how="left")
        else:
            wide[col_name] = pd.NA
        wide[col_name] = wide[col_name].fillna("none")

    wide["overall_sentiment"] = wide.apply(_calc_overall_sentiment, axis=1)

    return wide[_existing_columns(wide, WIDE_EXPORT_COLUMNS)]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    export_df = df[_existing_columns(df, WIDE_EXPORT_COLUMNS)] if not df.empty else df
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
