from __future__ import annotations

import re

import pandas as pd


CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_review_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_reviews(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"В таблице нет столбца {text_column}.")

    result = df.copy()
    result["processed_text"] = result[text_column].apply(clean_review_text)
    result["is_valid"] = result["processed_text"].str.len() > 0
    result["error_message"] = ""
    result.loc[~result["is_valid"], "error_message"] = "Пустой текст отзыва"
    return result

