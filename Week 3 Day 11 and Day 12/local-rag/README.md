# NovaTech Mock Document RAG agent

A local question answering system for documents, powered by Ollama, ChromaDB, and LangChain. It lets you ask natural language questions and get answers from your own PDFs and documents.

## Requirements

- Python 3.11+
- Ollama installed and running

## Setup

### 1. Install dependencies
Run the following command to install all necessary Python packages:
```bash
pip install openai chromadb langchain langchain-community pypdf
```

### 2. Pull required models

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 3. Add your documents

Place PDF files in the `documents/` folder.

## Usage

### Step 1

Run this once to chunk, embed, and store your documents in ChromaDB:

```bash
python ingest.py <- Ingests documents into ChromaDB
```

### Step 2
Query your documents with:
```bash
python pipeline.py "What is the remote work policy?"
```

### Step 3 Get answers

The system will return the most relevant answers from your document collection.


## Project Structure

```
local-rag/
├── config.py                   # All config in one place (swap LLM/embedding here)
├── ingest.py                   # Load PDFs, chunk, embed, store in ChromaDB
├── retrieve.py                 # Query ChromaDB, return ranked chunks
├── generate.py                 # Generate answer from chunks + query
├── pipeline.py                 # End-to-end pipeline
├── benchmark.py                # Model comparison script (Exercise 1)
├── benchmark_results.json      # Benchmark output
├── test_questions.json         # 18 Q&A pairs for evaluation
├── decision_log.md             # Technical decisions and reasoning
├── documents/                  # Source PDFs
└── chroma_db/                  # Vector DB (auto-created, gitignored)
```

## Models Used

| Role | Model |
|---|---|
| LLM | qwen3:8b |
| Embeddings | nomic-embed-text |
| Vector DB | ChromaDB (local, persistent) |