from __future__ import annotations

from io import BytesIO

import pandas as pd

from modules.aggregation import aggregate_results
from modules.data_loader import load_reviews, prepare_reviews_dataframe, selectable_text_columns
from modules.export import to_csv_bytes, to_xlsx_bytes
from modules.preprocessing import clean_review_text, preprocess_reviews


class UploadedFileStub:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_load_csv_and_create_review_id() -> None:
    source = UploadedFileStub("reviews.csv", "text,rating\nХороший товар,5\n".encode("utf-8"))
    df = load_reviews(source)
    prepared = prepare_reviews_dataframe(df, text_column="text")

    assert prepared.loc[0, "review_id"] == 1
    assert prepared.loc[0, "review_text"] == "Хороший товар"
    assert "product_name" in prepared.columns


def test_load_xlsx() -> None:
    buffer = BytesIO()
    pd.DataFrame({"review_text": ["Отлично"]}).to_excel(buffer, index=False)
    source = UploadedFileStub("reviews.xlsx", buffer.getvalue())

    df = load_reviews(source)

    assert df.loc[0, "review_text"] == "Отлично"


def test_selectable_text_columns_without_review_text() -> None:
    df = pd.DataFrame({"comment": ["Текст"], "rating": [5]})

    assert selectable_text_columns(df) == ["comment"]


def test_preprocess_marks_empty_reviews() -> None:
    df = pd.DataFrame({"review_text": ["  Нормально\n", "   ", None]})
    result = preprocess_reviews(df)

    assert clean_review_text("a\t b") == "a b"
    assert result.loc[0, "is_valid"] is True or bool(result.loc[0, "is_valid"])
    assert not bool(result.loc[1, "is_valid"])
    assert result.loc[1, "error_message"] == "Пустой текст отзыва"
    assert not bool(result.loc[2, "is_valid"])


def test_aggregate_results() -> None:
    results = pd.DataFrame(
        {
            "aspect": ["Упаковка", "Упаковка", "Доставка"],
            "sentiment": ["negative", "positive", "neutral"],
            "confidence": [0.9, 0.7, 0.6],
        }
    )

    agg = aggregate_results(results)
    packaging = agg[agg["aspect"] == "Упаковка"].iloc[0]

    assert packaging["mention_count"] == 2
    assert packaging["positive_count"] == 1
    assert packaging["negative_count"] == 1
    assert packaging["positive_share"] == 50.0
    assert packaging["negative_share"] == 50.0
    assert packaging["neutral_share"] == 0.0
    assert packaging["negative_rate"] == 50.0
    assert packaging["confidence_avg"] == 0.8


def test_aggregate_empty_contains_share_columns() -> None:
    agg = aggregate_results(pd.DataFrame())

    assert agg.empty
    assert "positive_share" in agg.columns
    assert "negative_share" in agg.columns
    assert "neutral_share" in agg.columns
    assert "negative_rate" in agg.columns


def test_exports_are_bytes() -> None:
    df = pd.DataFrame(
        {
            "review_id": [1],
            "review_text": ["Текст"],
            "aspect": ["Общее впечатление"],
            "sentiment": ["positive"],
            "sentiment_label": ["Положительная"],
            "confidence": [0.8],
        }
    )

    assert to_csv_bytes(df).startswith(b"\xef\xbb\xbf")
    assert to_xlsx_bytes(results_df=df)[:2] == b"PK"
