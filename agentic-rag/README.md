# agentic-rag

Production-ready RAG system for the **Agentic Design Patterns** PDF using:
- `pdfplumber` extraction
- Hybrid chunking (hierarchical + semantic + content-aware + bridge)
- Dense embeddings: local Ollama model `bge-large-en-v1.5:latest`
- Sparse retrieval: `BM25Encoder` (`pinecone-text`)
- Pinecone hybrid retrieval + RRF fusion + `BAAI/bge-reranker-large`
- Redis cache (30-day TTL)
- Gemini 2.5 Flash generation
- FastAPI serving
- RAGAS evaluation

## Project structure

```
agentic-rag/
├── ingestion/
├── retrieval/
├── generation/
├── cache/
├── api/
├── evaluation/
├── config/
├── tests/
├── docker-compose.yml
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

## 1) Setup

```bash
cd agentic-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create environment config:

```bash
copy .env.example .env
```

Set required values in `.env`:
- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `PDF_PATH` (full path to `Agentic Design Patterns` PDF)
- `OLLAMA_HOST` (default `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default `bge-large-en-v1.5:latest`)

## 2) Start Redis

```bash
docker compose up -d
```

## 3) Run ingestion pipeline

Runs in strict order: `extract -> chunk -> embed -> sparse -> upsert`

```bash
python -m ingestion.run_ingestion
```

## 4) Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `POST /query`
  - Request: `{"query":"...", "alpha":0.6}`
  - Response: `{"answer":"...", "citations":[...], "confidence":"...", "source":"cache|pipeline"}`
- `GET /health`
  - Response: `{"status":"ok","redis":"connected|disconnected","pinecone":"connected|disconnected"}`

## 5) Run tests

```bash
pytest -q
```

## 6) Run evaluation

```bash
python -m evaluation.evaluate
```

RAGAS metrics:
- `faithfulness`
- `answer_relevancy`
- `context_precision`
