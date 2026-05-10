from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.wb_parser import WBParserError, extract_nm_id, fetch_and_save_wb_reviews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собрать публичные текстовые отзывы Wildberries в CSV-файл."
    )
    parser.add_argument(
        "url",
        help="Ссылка на товар Wildberries или артикул nmId.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Максимальное число отзывов для сохранения. По умолчанию: 400.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Папка для CSV-файла. По умолчанию: data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        nm_id = extract_nm_id(args.url)
        df, path = fetch_and_save_wb_reviews(
            args.url,
            limit=args.limit,
            output_dir=args.output_dir,
        )
    except WBParserError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"Артикул WB: {nm_id}")
    print(f"Собрано текстовых отзывов: {len(df)}")
    print(f"CSV сохранён: {path}")
    print("Теперь этот файл можно загрузить в Streamlit-приложение.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
