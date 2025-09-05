# LLM Knowledge Extractor – Jouster (Take‑Home)

A tiny FastAPI service that accepts raw text, summarizes it, and extracts **structured metadata**:

- `title` (if provided)
- `topics` (3 key topics)
- `sentiment` (`positive` | `neutral` | `negative`)
- `keywords` (the 3 most frequent **nouns/terms** — implemented locally, *not* via LLM)
- Optional: `confidence` (naive heuristic)

Results are **persisted in SQLite** and searchable by topic/keyword.

> Timebox friendly: works out‑of‑the‑box with a local heuristic summarizer. If you set `USE_OPENAI=true` and provide `OPENAI_API_KEY`, it will call OpenAI for a 1–2 sentence summary. If the API fails, it falls back to the heuristic without crashing.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive Swagger.

### Environment (optional)

Create a `.env` file (or export env vars) if you want LLM summaries:

```
USE_OPENAI=true
OPENAI_API_KEY=sk-...
```

If `USE_OPENAI` is not set/true, a lightweight local summarizer is used.

---

## API

### `POST /analyze`

Analyze a single string or a list of strings.

**Request (single):**
```json
{ "text": "OpenAI released new tools. Developers are excited.", "title": "News blip" }
```

**Request (batch):**
```json
{ "texts": [
  "Python remains popular for data science.",
  "Go is appreciated for fast, static binaries."
]}
```

**Response:**
```json
{
  "items": [{
    "id": 1,
    "title": "News blip",
    "summary": "A 1–2 sentence summary...",
    "topics": ["openai", "tools", "developers"],
    "sentiment": "positive",
    "keywords": ["openai", "tools", "developers"],
    "confidence": 0.74,
    "created_at": "2025-01-01T12:00:00Z"
  }]
}
```

### `GET /search`

Query by `topic` or `keyword` (OR logic).

```
/search?topic=openai
/search?keyword=python
```

**Response** mirrors the stored analysis records.


# Example payloads & matching searches

### 1) Kubernetes cost controls
**POST**
```bash
curl -X POST http://127.0.0.1:8000/analyze   -H 'Content-Type: application/json'   -d '{
    "title": "Kubernetes cost controls",
    "text": "Kubernetes cluster costs can spike without guardrails. Use the cluster autoscaler and vertical pod autoscaler to right-size nodes. Spot instances reduce compute costs, while requests and limits prevent waste. Clear budgets, alerts, and a monthly cost review keep Kubernetes spending predictable."
  }'
```
**GET**
```bash
curl "http://127.0.0.1:8000/search?topic=kubernetes"
curl "http://127.0.0.1:8000/search?keyword=autoscaler"
curl "http://127.0.0.1:8000/search?keyword=costs"
```

### 2) Postgres vector search (pgvector)
**POST**
```bash
curl -X POST http://127.0.0.1:8000/analyze   -H 'Content-Type: application/json'   -d '{
    "title": "Postgres vector search",
    "text": "Postgres with the pgvector extension enables similarity search on embeddings. Using HNSW indexes speeds up queries, and batching writes helps maintain index performance. Many teams keep metadata in Postgres so pgvector and SQL live together, simplifying deployments."
  }'
```
**GET**
```bash
curl "http://127.0.0.1:8000/search?topic=postgres"
curl "http://127.0.0.1:8000/search?keyword=pgvector"
curl "http://127.0.0.1:8000/search?keyword=embeddings"
```

---

## Design Choices (3–5 sentences)

- **Simplicity first**: FastAPI + SQLite yields a small, self‑contained service that’s trivial to run and review. 
- **Abstraction over LLM**: `summarize_text()` tries OpenAI *iff* `USE_OPENAI=true`; otherwise it uses a deterministic heuristic that extracts the first sentence or two and compresses them. Errors are caught and we fall back gracefully.
- **Local NLP**: `extract_keywords()` implements term frequency filtering with a small stopword list and basic heuristics to approximate nouns, satisfying the “implement yourself” requirement without external models.
- **Searchability**: Topics/keywords are stored (as JSON) and indexed simply; `/search` matches rows containing the requested token(s).
- **Robustness**: Empty input returns a 422; LLM failures return a warning in the payload but never crash the server. Batch mode and a naive confidence score are included as bonuses.

### Trade‑offs

- No heavyweight NLP libraries are used (keeps setup fast) so noun detection is heuristic and may misclassify some words.
- SQLite is ideal for a take‑home; for multi‑user concurrency, a hosted Postgres would be preferable.
- Tests are minimal (focus on the pure functions) to respect the timebox.

---

## Tests

```bash
pytest -q

# If your environment needs an explicit path:
PYTHONPATH=. pytest -q
```

---

## Docker (optional)

```bash
docker build -t jouster .
docker run -p 8000:8000 --env-file .env jouster
```

---

## Notes

- Keywords are **not** extracted via the LLM; they are computed locally from term frequencies with heuristics to bias towards nouns/phrases.
