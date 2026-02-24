import os
import time
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

DOCUMENTS_DIR = "documents"
CHROMA_DB_DIR = "chroma_db"


def load_pdfs(documents_dir: str) -> list[dict]:
    """Load all PDFs from the documents directory."""
    docs = []
    pdf_files = list(Path(documents_dir).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{documents_dir}/'")

    for pdf_path in pdf_files:
        print(f"  Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            docs.append({
                "content": page.page_content,
                "source": pdf_path.name,
                "page": page.metadata.get("page", 0) + 1,
            })

    print(f"  Loaded {len(docs)} pages from {len(pdf_files)} PDFs")
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            if split.strip():
                chunks.append({
                    "content": split.strip(),
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_index": i,
                })

    print(f"  Created {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """Embed chunks and store in ChromaDB."""
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=config.LLM_API_KEY,
        api_base=config.EMBEDDING_BASE_URL,
        model_name=config.EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection to allow re-ingestion
    try:
        client.delete_collection(config.COLLECTION_NAME)
        print(f"  Deleted existing collection '{config.COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch insert
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["content"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]} for c in batch],
        )
        print(f"  Embedded chunks {i}–{i + len(batch) - 1}...")

    print(f"  Stored {collection.count()} chunks in ChromaDB")
    return collection


def ingest():
    print("=" * 50)
    print("Ingestion pipeline starting...")
    print("=" * 50)

    start = time.time()

    print("\n[1/3] Loading PDFs...")
    docs = load_pdfs(DOCUMENTS_DIR)

    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(docs)

    print("\n[3/3] Embedding and storing...")
    build_vector_store(chunks)

    elapsed = round(time.time() - start, 2)
    print(f"\nDone in {elapsed}s — ready to query.")


if __name__ == "__main__":
    ingest()