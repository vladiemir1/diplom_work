from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests


WB_REVIEW_ENDPOINTS = (
    "https://feedbacks1.wb.ru/feedbacks/v1/{nm_id}",
    "https://feedbacks2.wb.ru/feedbacks/v1/{nm_id}",
    "https://feedbacks1.wb.ru/feedbacks/v2/{nm_id}",
    "https://feedbacks2.wb.ru/feedbacks/v2/{nm_id}",
)


class WBParserError(RuntimeError):
    """Raised when public Wildberries reviews cannot be fetched."""


def extract_nm_id(value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise WBParserError("Введите ссылку на товар Wildberries или артикул.")

    catalog_match = re.search(r"/catalog/(\d+)", text)
    if catalog_match:
        return catalog_match.group(1)

    nm_match = re.search(r"(?:nmId|nmID|articul|article|артикул)[=/:\s]+(\d+)", text, re.I)
    if nm_match:
        return nm_match.group(1)

    digits = re.findall(r"\d{5,}", text)
    if len(digits) == 1:
        return digits[0]

    raise WBParserError("Не удалось извлечь артикул WB из ссылки.")


def _request_feedbacks(nm_id: str, timeout: int = 12) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": f"https://www.wildberries.ru/catalog/{nm_id}/detail.aspx",
    }
    errors: list[str] = []
    for template in WB_REVIEW_ENDPOINTS:
        url = template.format(nm_id=nm_id)
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                errors.append(f"{url}: HTTP {response.status_code}")
                continue
            data = response.json()
            if isinstance(data, dict) and "feedbacks" in data:
                return data
            errors.append(f"{url}: неожиданный формат ответа")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise WBParserError(
        "Не удалось получить публичные отзывы WB. "
        "Возможно, Wildberries ограничил endpoint или у товара нет доступных отзывов. "
        + " | ".join(errors[:3])
    )


def _extract_review_text(feedback: dict[str, Any]) -> str:
    parts = [
        str(feedback.get("text") or "").strip(),
        str(feedback.get("pros") or "").strip(),
        str(feedback.get("cons") or "").strip(),
    ]
    return " ".join(part for part in parts if part).strip()


def feedbacks_to_dataframe(data: dict[str, Any], nm_id: str, limit: int = 100) -> pd.DataFrame:
    feedbacks = data.get("feedbacks") or []
    rows = []
    for feedback in feedbacks[:limit]:
        text = _extract_review_text(feedback)
        if not text:
            continue
        rows.append(
            {
                "review_id": feedback.get("id") or len(rows) + 1,
                "product_name": f"Wildberries {nm_id}",
                "rating": feedback.get("productValuation"),
                "review_text": text,
                "date": feedback.get("createdDate"),
                "nm_id": nm_id,
                "wb_user_name": (feedback.get("wbUserDetails") or {}).get("name"),
            }
        )
    return pd.DataFrame(rows)


def fetch_wb_reviews(product_url: str, limit: int = 100) -> pd.DataFrame:
    nm_id = extract_nm_id(product_url)
    data = _request_feedbacks(nm_id)
    df = feedbacks_to_dataframe(data, nm_id=nm_id, limit=limit)
    if df.empty:
        raise WBParserError("У товара не найдено текстовых отзывов в публичном ответе WB.")
    return df


def save_wb_reviews_csv(df: pd.DataFrame, nm_id: str, output_dir: str | Path = "data") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"wb_reviews_{nm_id}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


def fetch_and_save_wb_reviews(
    product_url: str,
    limit: int = 100,
    output_dir: str | Path = "data",
) -> tuple[pd.DataFrame, Path]:
    nm_id = extract_nm_id(product_url)
    data = _request_feedbacks(nm_id)
    df = feedbacks_to_dataframe(data, nm_id=nm_id, limit=limit)
    if df.empty:
        raise WBParserError("У товара не найдено текстовых отзывов в публичном ответе WB.")
    return df, save_wb_reviews_csv(df, nm_id=nm_id, output_dir=output_dir)

