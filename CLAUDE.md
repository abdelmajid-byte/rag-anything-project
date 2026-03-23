# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fully local RAG (Retrieval-Augmented Generation) system for French legal/tax documents, combining:
- **LightRAG** knowledge graph for entity/relation-based retrieval
- **RAGAnything** for multimodal document parsing (tables, equations, images)
- **MinerU** as the PDF parser backend
- **Ollama** for all LLM, vision, and embedding inference (no cloud dependency)

## Setup & Running

### Prerequisites
Ollama must be running with these models pulled:
```bash
ollama pull llama3.1:8b
ollama pull llava:7b
ollama pull nomic-embed-text
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Start the API server
```bash
python api.py
```
API at `http://localhost:8000`, Swagger UI at `http://localhost:8000/docs`.

### Ingest documents
Place PDFs in `./donnees rag/`, then:
```bash
curl -X POST http://localhost:8000/ingest/folder
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Votre question ici", "mode": "hybrid"}'
```

Query modes: `naive` (pure vector), `local` (nearby graph entities), `global` (transversal themes), `hybrid` (local + global, recommended).

## Architecture

### Entry Points
- **`api.py`** — FastAPI server; all external interaction goes through here. Initializes the RAG pipeline at startup via lifespan context.
- **`src/ingestion/rag_anything_pipeline.py`** — Core engine; all RAG logic lives here.
- **`src/utils/helpers.py`** — Config loading (`load_config`), logging setup, directory creation.
- **`config.yaml`** — Single source of truth for all settings (models, paths, query defaults).

### Data Flow

**Ingestion:** PDF → MinerU parser → `content_list` (structured chunks with text/tables/images) → `separate_content()` (patched to filter structural noise) → LightRAG entity/relation extraction → embedding → knowledge graph persisted to `./data/rag_anything_storage/`.

**Query:** Question → LLM keyword extraction → hybrid vector + graph search → context assembly (≤1500 tokens) → response generation.

### Key Design Decisions
- **Context window budget:** Ollama's OpenAI-compatible endpoint caps context at ~2048 tokens. The pipeline allocates ~1500 tokens for retrieval context + ~500 for the prompt template. Chunk size is 400 tokens with 40-token overlap.
- **Custom French prompts:** `rag_anything_pipeline.py` monkey-patches LightRAG's default English prompts with French equivalents to prevent hallucinations on French documents.
- **`separate_content()` patch:** Overrides RAGAnything's method to convert structural elements (header, footer, list, page_number) to text-only, avoiding unnecessary LLM processing.
- **Single-worker parallelism:** Ollama runs on CPU locally; entity extraction uses 1 worker, no gleaning passes, and a 4000-token extraction context limit to avoid timeouts.
- **HuggingFace mirror:** Set on first run to avoid timeout issues downloading model files.

### Storage Layout
```
data/rag_anything_storage/    # LightRAG knowledge graph (JSON files)
donnees rag/                  # Source PDFs to ingest
output/                       # MinerU parser output (images, JSON)
logs/                         # Execution logs and evaluation reports
```

## Configuration

All settings in `config.yaml` can be overridden via environment variables (see `.env.example`). Key settings:
- `ollama.host` — Ollama API URL (default: `http://localhost:11434`)
- `ollama.llm_model` — Text generation model (default: `llama3.1:8b`)
- `ollama.vision_model` — Image analysis model (default: `llava:7b`)
- `ollama.embedding_model` — Embedding model (default: `nomic-embed-text`, 768d)
- `parser.backend` — Document parser (`mineru`, `docling`, `paddleocr`)
- `parser.device` — Compute device (`cpu`, `cuda`)
- `query.default_mode` — Default retrieval mode (`hybrid`)

## Testing

- **`test_questions.yaml`** — 10 French tax/invoicing questions with ground truth, keywords, and recommended query mode.
- **`postman_collection.json`** — Postman collection with all API endpoints pre-configured.
- **`/health` endpoint** — Checks Ollama connectivity and RAG index status before running queries.

To clear the LLM response cache (useful when debugging): delete or empty `data/rag_anything_storage/kv_store_llm_response_cache.json`.
