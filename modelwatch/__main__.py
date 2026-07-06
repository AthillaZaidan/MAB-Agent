from __future__ import annotations

import argparse

from modelwatch.config import load_config
from modelwatch.connectors import default_connectors
from modelwatch.extractor import OllamaExtractor
from modelwatch.judge import OllamaJudge
from modelwatch.pipeline import run_pipeline
from modelwatch.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m modelwatch")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--window-hours", type=int, default=48)
    run.add_argument("--config")
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--config", default="modelwatch.config.json")
    args = parser.parse_args()

    if args.command == "schedule":
        print(f"0 1 * * * cd $PWD && python -m modelwatch run --window-hours 48 --config {args.config}")
        return

    config = load_config(args.config)
    result = run_pipeline(
        connectors=default_connectors(config),
        extractor=OllamaExtractor(config.ollama_url, config.ollama_model),
        judge=OllamaJudge(config.ollama_url, config.ollama_model),
        store=Store(config.database_path),
        output_dir=config.output_dir,
        window_hours=args.window_hours,
        log=lambda message: print(message, flush=True),
    )
    print(
        f"{result.status}: {result.source_count} source items, "
        f"{result.candidate_count} candidates, digest={result.digest_path}"
    )


if __name__ == "__main__":
    main()
