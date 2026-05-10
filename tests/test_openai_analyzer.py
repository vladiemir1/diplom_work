from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from modules.openai_analyzer import (
    AnalyzerConfig,
    AnalyzerError,
    analyze_reviews,
    get_gigachat_access_token,
    is_gigachat_config,
    load_api_key,
)


class ResponsesStub:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class ClientStub:
    def __init__(self, payload: dict):
        self.chat = SimpleNamespace(completions=ResponsesStub(payload))


def test_analyze_reviews_with_mocked_openai_client() -> None:
    df = pd.DataFrame(
        {
            "review_id": [1],
            "product_name": ["Товар"],
            "rating": [2],
            "date": [pd.NaT],
            "review_text": ["Товар хороший, но упаковка мятая"],
            "processed_text": ["Товар хороший, но упаковка мятая"],
        }
    )
    client = ClientStub(
        {
            "items": [
                {
                    "review_id": "1",
                    "results": [
                        {
                            "aspect": "Качество товара",
                            "sentiment": "positive",
                            "confidence": 0.86,
                            "explanation": "Покупатель положительно оценивает товар.",
                        },
                        {
                            "aspect": "Упаковка",
                            "sentiment": "negative",
                            "confidence": 0.91,
                            "explanation": "Указана мятая упаковка.",
                        },
                    ],
                }
            ]
        }
    )

    result = analyze_reviews(df, client=client, config=AnalyzerConfig(batch_size=5))

    assert len(result) == 2
    assert set(result["aspect"]) == {"Качество товара", "Упаковка"}
    assert "json_schema" in json.dumps(client.chat.completions.calls[0], ensure_ascii=False)


def test_analyze_reviews_adds_general_aspect_when_model_returns_empty_results() -> None:
    df = pd.DataFrame(
        {
            "review_id": [1],
            "product_name": [pd.NA],
            "rating": [5],
            "date": [pd.NaT],
            "review_text": ["Всё супер"],
            "processed_text": ["Всё супер"],
        }
    )
    client = ClientStub({"items": [{"review_id": "1", "results": []}]})

    result = analyze_reviews(df, client=client)

    assert result.loc[0, "aspect"] == "Общее впечатление"
    assert result.loc[0, "sentiment_label"] == "Нейтральная"


def test_missing_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("modules.openai_analyzer.os.getenv", lambda key, default="": "")

    with pytest.raises(AnalyzerError):
        load_api_key()


def test_gigachat_config_detection() -> None:
    assert is_gigachat_config(AnalyzerConfig(provider="gigachat"))
    assert is_gigachat_config(AnalyzerConfig(model="GigaChat"))
    assert is_gigachat_config(AnalyzerConfig(base_url="https://gigachat.devices.sberbank.ru/api/v1"))


def test_get_gigachat_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class ResponseStub:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"access_token": "token-123"}

    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return ResponseStub()

    monkeypatch.setattr("modules.openai_analyzer.requests.post", fake_post)
    monkeypatch.delenv("GIGACHAT_ACCESS_TOKEN", raising=False)

    token = get_gigachat_access_token("authorization-key", verify_ssl=False)

    assert token == "token-123"
    assert calls[0][1]["headers"]["Authorization"] == "Basic authorization-key"
    assert calls[0][1]["data"]["scope"] == "GIGACHAT_API_PERS"
