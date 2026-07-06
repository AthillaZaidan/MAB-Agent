# MAB-Agent

ModelWatch is a local-first model release intelligence pipeline. It monitors
trusted model sources, filters weak signals with a local LLM judge, extracts
structured metadata, stores evidence, retrieves RAG snippets, scores benchmark
relevance, writes a timestamped Markdown digest, and can email the result.

## Architecture

```text
API/RSS sources
-> raw SQLite evidence
-> evidence chunking
-> Ollama embeddings
-> SQLite vector store
-> Ollama judge keep/reject
-> Ollama metadata extractor
-> dedupe + scoring
-> timestamped Markdown digest
-> optional SMTP email
```

The system is bounded: it does not let an LLM browse the web freely. Source
collection is deterministic, while Ollama is used for judging, extraction, and
embedding evidence.

## Prerequisites

- Python 3.11+ in the `base` conda environment
- Ollama running locally
- Local models:

```bash
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

If `nomic-embed-text` is missing, the run still completes, but RAG snippets are
disabled for that run.

## Configuration

Main config lives in [modelwatch.config.example.json](/home/athilla/Documents/IF_ITB/Personal-Project/MAB-Agent/modelwatch.config.example.json).

Important fields:

```json
{
  "output_dir": "data/digests",
  "database_path": "data/modelwatch.sqlite",
  "vector_database_path": "data/vectors.sqlite",
  "ollama_url": "http://localhost:11434",
  "ollama_model": "qwen3:4b-instruct",
  "ollama_embedding_model": "nomic-embed-text",
  "email_to": "athillazaidanstudy@gmail.com",
  "max_items_per_source": 100
}
```

## Email

Copy the tracked template, then edit the local file:

```bash
cp .env.example .env
```

For Gmail, use an App Password, not your normal login password:

```bash
MODELWATCH_SMTP_HOST=smtp.gmail.com
MODELWATCH_SMTP_PORT=587
MODELWATCH_SMTP_USERNAME=athillazaidanstudy@gmail.com
MODELWATCH_SMTP_PASSWORD=replace-with-gmail-app-password
MODELWATCH_EMAIL_FROM='ModelWatch <athillazaidanstudy@gmail.com>'
```

`.env` is ignored by git. Do not commit real credentials.

## Run

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
set -a
source .env
set +a
python -m modelwatch run --window-hours 168 --config modelwatch.config.example.json
```

Common windows:

```bash
# last 48 hours
python -m modelwatch run --window-hours 48 --config modelwatch.config.example.json

# last 7 days
python -m modelwatch run --window-hours 168 --config modelwatch.config.example.json

# last 30 days
python -m modelwatch run --window-hours 720 --config modelwatch.config.example.json
```

## Outputs

The run writes:

- `data/modelwatch.sqlite`: source items, candidates, run history
- `data/vectors.sqlite`: local vector evidence store
- `data/digests/digest-YYYY-MM-DD-HHMMSS.md`: timestamped digest

Digest files include:

- summary counts
- benchmark/watchlist/store-only sections
- evidence URLs
- RAG evidence snippets when embeddings are available
- connector failures and warnings

## Scheduling

Print the cron line:

```bash
python -m modelwatch schedule --config modelwatch.config.example.json
```

Production cron should load conda and `.env`, for example:

```cron
0 1 * * * cd /home/athilla/Documents/IF_ITB/Personal-Project/MAB-Agent && source "$($HOME/anaconda3/bin/conda info --base)/etc/profile.d/conda.sh" && conda activate base && set -a && source .env && set +a && python -m modelwatch run --window-hours 48 --config modelwatch.config.example.json >> data/modelwatch.log 2>&1
```

## Troubleshooting

Embedding model missing:

```text
[rag] ... embedding model unavailable; run `ollama pull nomic-embed-text`
```

Fix:

```bash
ollama pull nomic-embed-text
```

Email credential missing:

```text
[email] failed: RuntimeError: Missing email environment variable: MODELWATCH_SMTP_HOST
```

Fix: load `.env` before running.

RSS 404 or timeout:

```text
[source] rss warning: ... HTTP Error 404
```

The run continues. Update `rss_urls` if the provider changed its feed URL.

No OpenRouter/closed-source models in digest:

The current connector depends on the OpenRouter catalog response and current
window. The robust next step is catalog snapshot diffing so proprietary models
do not disappear just because they are outside the latest page.

## Tests

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
python -m pytest -q
```

## Production Notes

- Keep `.env` private.
- Keep `data/` persistent between runs; SQLite history is part of the product.
- Run with a stable Ollama service.
- Start with `--window-hours 48` for daily operation.
- Use larger windows like `168` or `720` for backfill, but expect slower Ollama judging.
- Back up `data/modelwatch.sqlite` and `data/vectors.sqlite` if the digest history matters.
