import time
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import config

CHROMA_DB_DIR = "chroma_db"


def get_collection() -> chromadb.Collection:
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=config.LLM_API_KEY,
        api_base=config.EMBEDDING_BASE_URL,
        model_name=config.EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def retrieve_chunks(query: str, top_k: int = config.TOP_K) -> dict:
    """
    Query the vector store and return top-k relevant chunks.

    Returns:
        {
            "query": str,
            "retrieved_chunks": [{"content", "source", "page", "relevance_score"}, ...],
            "retrieval_time_ms": int,
        }
    """
    start = time.time()

    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    elapsed_ms = round((time.time() - start) * 1000)

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score 0–1
        relevance_score = round(1 - (distance / 2), 4)
        chunks.append({
            "content": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", None),
            "relevance_score": relevance_score,
        })

    return {
        "query": query,
        "retrieved_chunks": chunks,
        "retrieval_time_ms": elapsed_ms,
    }


if __name__ == "__main__":
    import json
    query = input("Enter a test query: ")
    result = retrieve_chunks(query)
    print(json.dumps(result, indent=2))