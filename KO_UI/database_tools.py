#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "instance" / "ko.sqlite3"


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"DB 파일이 없습니다: {path}\n먼저 UI를 한 번 실행해 주세요.")
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def status(path: Path) -> None:
    with connect(path) as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        print(f"DB: {path}")
        for table in tables:
            count = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"- {table}: {count}건")


def show_rows(path: Path, table: str, limit: int) -> None:
    allowed = {"users", "training_sessions", "vision_results", "ai_reports"}
    if table not in allowed:
        raise SystemExit(f"지원하지 않는 테이블입니다: {table}")
    with connect(path) as db:
        rows = [dict(row) for row in db.execute(f'SELECT * FROM "{table}" ORDER BY id DESC LIMIT ?', (limit,))]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def export_csv(path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        for table in ("users", "training_sessions", "vision_results", "ai_reports"):
            rows = db.execute(f'SELECT * FROM "{table}" ORDER BY id').fetchall()
            if not rows:
                continue
            target = output / f"{table}.csv"
            with target.open("w", newline="", encoding="utf-8-sig") as fp:
                writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(dict(row) for row in rows)
            print(f"저장: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KO SQLite 확인 도구")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="테이블별 저장 건수를 확인합니다.")
    show = sub.add_parser("show", help="테이블 내용을 JSON으로 확인합니다.")
    show.add_argument("table", choices=["users", "training_sessions", "vision_results", "ai_reports"])
    show.add_argument("--limit", type=int, default=20)
    export = sub.add_parser("export", help="테이블을 CSV로 내보냅니다.")
    export.add_argument("--output", type=Path, default=BASE_DIR / "exports")
    args = parser.parse_args()

    if args.command == "status":
        status(args.database)
    elif args.command == "show":
        show_rows(args.database, args.table, max(1, min(1000, args.limit)))
    elif args.command == "export":
        export_csv(args.database, args.output)


if __name__ == "__main__":
    main()
