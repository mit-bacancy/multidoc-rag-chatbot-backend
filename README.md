# multidoc-rag backend (FastAPI)

Chat API over three pre-ingested reports. Each document/strategy pair has its own PGVector collection in Supabase, so retrieval never mixes reports.

| doc_id       | Document                        | Strategy                         |
|--------------|---------------------------------|----------------------------------|
| `gpo`        | GPO Annual Report               | `hierarchical` (+ `basic` baseline) |
| `worldbank`  | World Bank Economic Report      | `hierarchical`                   |
| `apple_imac` | Apple iMac Environmental Report | `multimodal`                     |

The UI lives in the separate frontend repo (`multidoc-rag-frontend`, Next.js).

## Prerequisites

- Python 3.12 (3.10+ works) and [uv](https://docs.astral.sh/uv/) (or plain `pip`)
- A Supabase project (any Postgres 15+ with the `pgvector` extension also works). Copy the connection string from *Project Settings → Database*; the session pooler URL is fine.
- A Google AI Studio API key for Gemini (`GOOGLE_API_KEY`)
- Node.js 20.9+ for the frontend

## Setup

```bash
uv venv -p 3.12 .venv
source .venv/bin/activate
# CPU-only torch keeps the venv small; drop the extra index on a GPU machine
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
cp .env.example .env
```

| Variable          | Required | Notes |
|-------------------|----------|-------|
| `SUPABASE_DB_URL` | yes      | `postgresql://` or `postgresql+psycopg://`; the driver prefix is added if missing |
| `GOOGLE_API_KEY`  | yes      | Gemini for answers and figure captions |
| `FRONTEND_ORIGIN` | no       | CORS origin, default `http://localhost:3000` |
| `LLM_MODEL`, `VISION_MODEL` | no | default `gemini-3.1-flash-lite` |
| `HF_TOKEN`        | no       | avoids Hugging Face rate limits when downloading the embedding/rerank models |

The first run downloads the embedding and reranker models from Hugging Face, so expect a few minutes.

## Full stack quickstart

```bash
# 1. backend (this repo)
source .venv/bin/activate
python -m scripts.init_db
python -m ingest.run_ingest --doc gpo        # repeat for worldbank, apple_imac
uvicorn app.main:app --reload --port 8000

# 2. frontend (other repo, second terminal)
cd ../frontend
npm install
cp .env.example .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                   # http://localhost:3000
```

If the frontend runs on another origin, set `FRONTEND_ORIGIN` here to match or the browser will block requests (CORS).

## Ingest (development time only)

PDFs are not committed. Put them in `data/pdfs/` with the filenames set in `DOCUMENTS` (`app/config.py`): `GPO-Anual-report.pdf`, `GEP-Jun-2026.pdf`, `iMac_PER_Oct2024.pdf`. Then:

```bash
python -m scripts.check_pdfs          # page counts, text layer, image/table pages
python -m scripts.init_db             # pgvector + parent_chunks + ingest_runs
python -m ingest.run_ingest --doc gpo           # hierarchical + basic baseline
python -m ingest.run_ingest --doc worldbank
python -m ingest.run_ingest --doc apple_imac    # captions figures with Gemini vision
```

Flags: `--strategy basic|hierarchical|multimodal`, `--rebuild` (re-ingest an already recorded pair), `--reparse` (ignore the cached Docling parse in `data/parsed/`), `--ocr` (scanned PDFs).

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

| Method | Route        | Notes |
|--------|--------------|-------|
| GET    | `/health`    | liveness + DB check |
| GET    | `/documents` | `[{doc_id, title, pages, ready}]` |
| POST   | `/chat`      | `{doc_id, message, history[], strategy?}` → `{answer, sources[]}` |

## Evaluate

Add questions to `eval/questions/<doc_id>.json` (format in `eval/run_eval.py`), then:

```bash
python -m eval.run_eval --doc gpo     # basic vs hierarchical side by side
python -m eval.run_eval --doc apple_imac
```

Reports go to `eval/results/<doc_id>_<date>.md`.

## Layout

```
app/        config (DOCUMENTS registry), store, retrieval (vector+BM25 → RRF → rerank → parents), generation, API
ingest/     parse (Docling → cached JSON), strategies/{basic,hierarchical,multimodal}, vision (figure captions)
eval/       questions + runner
scripts/    check_pdfs, init_db
```

## Eval results (2026-10-12)

All nine sample questions from the task brief are answered correctly with exact page citations. Full answers and retrieved sections are in `eval/results/`. No expected answers or pages were supplied, so hit@k and judge scores are blank; add `expected_answer` / `expected_pages` to the question files to get them.

| Doc | Question | Answer | Retrieved from |
|---|---|---|---|
| gpo | Depository libraries visited | 166 (in 25 states) | p. 5, p. 21 |
| gpo | FDsys replacement | govinfo; out of beta Dec 2017, FDsys retired Dec 2018 | p. 9, p. 22 |
| gpo | Procurement for agencies | $375.7 million | p. 5, p. 17 |
| worldbank | South Asia outlook | 6.3% in 2026, ~7% avg 2027–28 | Ch. 2 > South Asia > Outlook |
| worldbank | Chapter 3 on sovereign debt | debt >70% of GDP, debt service 11% of revenue, spreads +110 bp | Ch. 3 intro + summary |
| worldbank | Exec summary risks | conflict/commodity disruptions, energy+financial stress (1.3% growth), trade uncertainty | Executive Summary p. 19 |
| apple_imac | Energy vs ENERGY STAR | 58% less | p. 7 (text + endnote 18) |
| apple_imac | Carbon footprint split | 346 kg CO2e; production 50%, use 45% | p. 10 table |
| apple_imac | Stand material | 100% recycled aluminum | p. 4 icon call-out |

Task 1: `basic` and `hierarchical` both answer the GPO questions; `basic` cites less precisely and has no section context.

Notes from ingestion:
- Docling reports flat heading levels, so levels come from font size relative to body text (`ingest/parse.py: enrich`, `hierarchical.prepare_elements`). GEP chapters come from the PDF outline (`use_outline` in `DOCUMENTS`).
- Apple footnotes are endnotes on pp. 11–12; they are linked to the passages that cite their markers.
- Vision describes whole visual pages (vector charts are not detected as pictures); results are cached in `data/parsed/<doc_id>.pages.json`.
