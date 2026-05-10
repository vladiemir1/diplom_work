from __future__ import annotations

import pandas as pd
import pytest

from modules import wb_parser
from modules.wb_parser import (
    WBParserError,
    extract_nm_id,
    feedbacks_to_dataframe,
    fetch_and_save_wb_reviews,
)


def test_extract_nm_id_from_wb_link() -> None:
    assert (
        extract_nm_id("https://www.wildberries.ru/catalog/5870243/detail.aspx")
        == "5870243"
    )
    assert extract_nm_id("5870243") == "5870243"


def test_feedbacks_to_dataframe() -> None:
    data = {
        "feedbacks": [
            {
                "id": "abc",
                "text": "Хороший товар",
                "pros": "Быстрая доставка",
                "cons": "",
                "productValuation": 5,
                "createdDate": "2025-01-01T00:00:00Z",
                "wbUserDetails": {"name": "Анна"},
            }
        ]
    }

    df = feedbacks_to_dataframe(data, nm_id="123", limit=10)

    assert df.loc[0, "review_id"] == "abc"
    assert "Быстрая доставка" in df.loc[0, "review_text"]
    assert df.loc[0, "rating"] == 5


def test_fetch_and_save_wb_reviews_with_mock(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        wb_parser,
        "_request_feedbacks",
        lambda nm_id: {
            "feedbacks": [
                {
                    "id": "abc",
                    "text": "Упаковка отличная",
                    "productValuation": 5,
                    "createdDate": "2025-01-01T00:00:00Z",
                    "wbUserDetails": {},
                }
            ]
        },
    )

    df, path = fetch_and_save_wb_reviews(
        "https://www.wildberries.ru/catalog/123456/detail.aspx",
        output_dir=tmp_path,
    )

    assert isinstance(df, pd.DataFrame)
    assert path.exists()
    assert path.name == "wb_reviews_123456.csv"


def test_fetch_and_save_wb_reviews_empty(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(wb_parser, "_request_feedbacks", lambda nm_id: {"feedbacks": []})

    with pytest.raises(WBParserError):
        fetch_and_save_wb_reviews("https://www.wildberries.ru/catalog/123456/detail.aspx", output_dir=tmp_path)

