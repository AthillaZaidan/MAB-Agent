# MAB-Agent

Local MVP for ModelWatch, a model release intelligence pipeline.

## Run

Use the base conda environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
python -m modelwatch run --window-hours 48 --config modelwatch.config.example.json
```

The run writes:

- SQLite state to `data/modelwatch.sqlite`
- Markdown digest to `data/digests/digest-YYYY-MM-DD.md`

The extractor calls Ollama at `http://localhost:11434` and defaults to
`qwen3:4b-instruct`.

The run uses a bounded AI judge before extraction:

```text
API/RSS source -> broad candidates -> Ollama judge keep/reject -> extract kept items -> digest
```

This avoids hardcoding a fixed model list while still rejecting random LoRAs,
test uploads, and weak fine-tunes before they reach the digest.

## Schedule

Print a cron entry for the daily 01.00 run:

```bash
python -m modelwatch schedule --config modelwatch.config.example.json
```

## Test

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
python -m pytest -q
```

## Scope

This is intentionally small: stdlib HTTP connectors, SQLite, Ollama JSON
extraction, dedupe, scoring, and markdown output. Slack, LanceDB, dashboards,
and automatic benchmarking are left out until the local digest proves useful.
