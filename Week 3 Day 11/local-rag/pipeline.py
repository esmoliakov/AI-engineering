import json
from retrieve import retrieve_chunks
from generate import generate_answer
import sys

def answer_question(query: str) -> dict:
    retrieval_result = retrieve_chunks(query)
    generation_result = generate_answer(
        query=query,
        chunks=retrieval_result["retrieved_chunks"],
    )
    return {
        **generation_result,
        "retrieval": retrieval_result,
    }


if __name__ == "__main__":

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    print(f"\nQuery: {query}\n")

    result = answer_question(query)

    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources used:")
    for chunk in result["retrieval"]["retrieved_chunks"]:
        print(f"  - {chunk['source']} p.{chunk['page']} (score: {chunk['relevance_score']})")
    print(f"\nRetrieval: {result['retrieval']['retrieval_time_ms']}ms | Generation: {result['generation_time_ms']}ms")