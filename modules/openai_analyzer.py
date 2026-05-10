from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
import requests


GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_DEFAULT_SCOPE = "GIGACHAT_API_PERS"


ASPECTS = [
    "Качество товара",
    "Соответствие описанию",
    "Упаковка",
    "Доставка",
    "Общее впечатление",
]

SENTIMENTS = ["positive", "negative", "neutral"]
SENTIMENT_LABELS = {
    "positive": "Положительная",
    "negative": "Отрицательная",
    "neutral": "Нейтральная",
}


class AnalyzerError(RuntimeError):
    """Raised when the external NLP module cannot complete analysis."""


@dataclass(frozen=True)
class AnalyzerConfig:
    model: str = "gpt-4o-mini"
    batch_size: int = 10
    temperature: float = 0.0
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    provider: str | None = None


def load_api_key(explicit_api_key: str | None = None) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise AnalyzerError(
            "Не найден API-ключ. Укажите LLM_API_KEY/OPENAI_API_KEY в .env "
            "или введите ключ в интерфейсе."
        )
    return api_key


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def is_gigachat_config(config: AnalyzerConfig) -> bool:
    provider = (config.provider or "").lower()
    base_url = (config.base_url or "").lower()
    model = (config.model or "").lower()
    return provider == "gigachat" or "gigachat" in base_url or model.startswith("gigachat")


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "нет"}


def _normalize_gigachat_authorization(value: str) -> str:
    stripped = value.strip()
    if stripped.lower().startswith(("basic ", "bearer ")):
        return stripped
    return f"Basic {stripped}"


def get_gigachat_access_token(
    credentials: str | None = None,
    scope: str | None = None,
    verify_ssl: bool | None = None,
) -> str:
    _load_env()
    access_token = (os.getenv("GIGACHAT_ACCESS_TOKEN") or "").strip()
    if access_token and not credentials:
        return access_token

    auth_key = (
        credentials
        or os.getenv("GIGACHAT_CREDENTIALS")
        or os.getenv("GIGACHAT_AUTHORIZATION_KEY")
        or ""
    ).strip()
    if not auth_key:
        raise AnalyzerError(
            "Для GigaChat нужен Authorization Key. Укажите GIGACHAT_CREDENTIALS "
            "в .env или введите ключ в расширенных настройках."
        )

    verify = _env_bool("GIGACHAT_VERIFY_SSL", True) if verify_ssl is None else verify_ssl
    try:
        response = requests.post(
            GIGACHAT_OAUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": _normalize_gigachat_authorization(auth_key),
            },
            data={"scope": scope or os.getenv("GIGACHAT_SCOPE") or GIGACHAT_DEFAULT_SCOPE},
            timeout=20,
            verify=verify,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise AnalyzerError(f"Не удалось получить access token GigaChat: {exc}") from exc

    token = str(data.get("access_token") or "").strip()
    if not token:
        raise AnalyzerError("GigaChat OAuth не вернул access_token.")
    return token


def build_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "review_id": {"type": "string"},
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "aspect": {"type": "string", "enum": ASPECTS},
                                    "sentiment": {"type": "string", "enum": SENTIMENTS},
                                    "confidence": {"type": "number"},
                                    "explanation": {"type": "string"},
                                },
                                "required": [
                                    "aspect",
                                    "sentiment",
                                    "confidence",
                                    "explanation",
                                ],
                            },
                        },
                    },
                    "required": ["review_id", "results"],
                },
            }
        },
        "required": ["items"],
    }


SYSTEM_PROMPT = """
Ты модуль аспектного анализа отзывов маркетплейса.
Верни только структурированный JSON по заданной схеме.

Задача:
1. Для каждого отзыва найди один или несколько аспектов.
2. Используй только эти аспекты: Качество товара, Соответствие описанию, Упаковка, Доставка, Общее впечатление.
3. Если в отзыве нет конкретного аспекта, используй Общее впечатление.
4. Для каждого аспекта определи тональность: positive, negative или neutral.
5. Учитывай rating как дополнительный сигнал, но не заменяй им смысл текста.
6. confidence указывай числом от 0 до 1.
7. explanation пиши кратко, по-русски, одной фразой до 140 символов.
8. Один отзыв может иметь несколько результатов, если в тексте несколько аспектов.
""".strip()


def _build_client(config: AnalyzerConfig) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AnalyzerError(
            "Пакет openai не установлен. Выполните: pip install -r requirements.txt"
        ) from exc

    if is_gigachat_config(config):
        api_key = get_gigachat_access_token(config.api_key)
        base_url = config.base_url or os.getenv("GIGACHAT_BASE_URL") or GIGACHAT_BASE_URL
    else:
        api_key = load_api_key(config.api_key)
        base_url = config.base_url

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if is_gigachat_config(config) and not _env_bool("GIGACHAT_VERIFY_SSL", True):
        try:
            import httpx

            kwargs["http_client"] = httpx.Client(verify=False)
        except Exception:
            pass
    return OpenAI(**kwargs)


def _extract_response_text(response: Any) -> str:
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if content:
            return str(content)

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    if isinstance(response, dict):
        if response.get("output_text"):
            return str(response["output_text"])
        output = response.get("output", [])
    else:
        output = getattr(response, "output", [])

    chunks: list[str] = []
    for item in output or []:
        content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for part in content or []:
            if isinstance(part, dict):
                text = part.get("text") or part.get("output_text")
            else:
                text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))

    if chunks:
        return "".join(chunks)

    raise AnalyzerError("OpenAI API вернул ответ без текстового содержимого.")


def _completion_create(client: Any, **kwargs: Any) -> Any:
    return client.chat.completions.create(**kwargs)


def _call_openai_batch(
    records: list[dict[str, Any]],
    client: Any,
    config: AnalyzerConfig,
) -> dict[str, Any]:
    payload = {
        "reviews": records,
        "output_contract": {
            "review_id": "string",
            "results": [
                {
                    "aspect": ASPECTS,
                    "sentiment": SENTIMENTS,
                    "confidence": "number 0..1",
                    "explanation": "short Russian explanation",
                }
            ],
        },
    }

    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Проанализируй отзывы. JSON входных данных:\n"
                + json.dumps(payload, ensure_ascii=False),
            },
        ]

    strict_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "review_analysis_batch",
            "schema": build_response_schema(),
            "strict": True,
        },
    }

    try:
        response = _completion_create(
            client,
            model=config.model,
            messages=messages,
            response_format=strict_response_format,
            temperature=config.temperature,
        )
    except Exception:
        # Many OpenAI-compatible providers support JSON mode before strict
        # JSON Schema. The schema is still present in the prompt payload.
        try:
            response = _completion_create(
                client,
                model=config.model,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": "Верни валидный JSON без Markdown и без пояснений вне JSON.",
                    }
                ],
                response_format={"type": "json_object"},
                temperature=config.temperature,
            )
        except Exception as exc:
            raise AnalyzerError(f"LLM API недоступен или отклонил запрос: {exc}") from exc

    try:
        return json.loads(_extract_response_text(response))
    except json.JSONDecodeError as exc:
        raise AnalyzerError("LLM API вернул некорректный JSON.") from exc


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _normalize_model_items(raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise AnalyzerError("Ответ модели не соответствует ожидаемой структуре.")

    raw_items = raw.get("items")
    if raw_items is None:
        raw_items = raw.get("reviews")
    if not isinstance(raw_items, list):
        raise AnalyzerError("Ответ модели не соответствует ожидаемой структуре.")

    normalized: dict[str, list[dict[str, Any]]] = {}
    for item in raw_items:
        review_id = str(item.get("review_id", ""))
        results = item.get("results") or []
        normalized_results = []
        for result in results:
            raw_aspect = result.get("aspect")
            aspects = raw_aspect if isinstance(raw_aspect, list) else [raw_aspect]
            sentiment = result.get("sentiment")
            if sentiment not in SENTIMENTS:
                sentiment = "neutral"
            for aspect in aspects:
                if aspect not in ASPECTS:
                    aspect = "Общее впечатление"
                normalized_results.append(
                    {
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "sentiment_label": SENTIMENT_LABELS[sentiment],
                        "confidence": _clamp_confidence(result.get("confidence")),
                        "explanation": str(result.get("explanation") or "").strip(),
                    }
                )
        normalized[review_id] = normalized_results or [
            {
                "aspect": "Общее впечатление",
                "sentiment": "neutral",
                "sentiment_label": SENTIMENT_LABELS["neutral"],
                "confidence": 0.3,
                "explanation": "Модель не выделила конкретный аспект.",
            }
        ]
    return normalized


def _reviews_to_records(df: pd.DataFrame, text_column: str) -> list[dict[str, Any]]:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "review_id": str(row["review_id"]),
                "text": str(row[text_column]),
                "rating": None if pd.isna(row.get("rating")) else row.get("rating"),
                "product_name": None
                if pd.isna(row.get("product_name"))
                else str(row.get("product_name")),
            }
        )
    return records


def analyze_reviews(
    df: pd.DataFrame,
    text_column: str = "processed_text",
    client: Any | None = None,
    config: AnalyzerConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
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
        )
    if text_column not in df.columns:
        raise AnalyzerError(f"В таблице нет столбца для анализа: {text_column}.")

    config = config or AnalyzerConfig()
    client = client or _build_client(config)
    records = _reviews_to_records(df, text_column)
    analyzed: dict[str, list[dict[str, Any]]] = {}

    for start in range(0, len(records), config.batch_size):
        batch = records[start : start + config.batch_size]
        raw = _call_openai_batch(batch, client, config)
        analyzed.update(_normalize_model_items(raw))
        if progress_callback:
            progress_callback(min(start + len(batch), len(records)), len(records))

    rows = []
    for _, row in df.iterrows():
        review_id = str(row["review_id"])
        row_results = analyzed.get(review_id) or [
            {
                "aspect": "Общее впечатление",
                "sentiment": "neutral",
                "sentiment_label": SENTIMENT_LABELS["neutral"],
                "confidence": 0.3,
                "explanation": "Модель не вернула результат по этому отзыву.",
            }
        ]
        for result in row_results:
            rows.append(
                {
                    "review_id": row.get("review_id"),
                    "product_name": row.get("product_name"),
                    "rating": row.get("rating"),
                    "date": row.get("date"),
                    "review_text": row.get("review_text"),
                    "processed_text": row.get("processed_text"),
                    **result,
                }
            )

    return pd.DataFrame(rows)
