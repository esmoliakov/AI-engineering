import time
from openai import OpenAI

import config


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer from retrieved chunks.

    Args:
        query: The user's question.
        chunks: List of retrieved chunk dicts (from retrieve.py).

    Returns:
        {
            "query": str,
            "answer": str,
            "context_used": str,
            "model": str,
            "generation_time_ms": int,
            "tokens_generated": int,
        }
    """
    # Build context string from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']} (page {chunk['page']})\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = config.SYSTEM_PROMPT.format(context=context)

    client = OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )

    start = time.time()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
    )
    elapsed_ms = round((time.time() - start) * 1000)

    answer = response.choices[0].message.content
    usage = response.usage

    return {
        "query": query,
        "answer": answer,
        "context_used": context,
        "model": config.LLM_MODEL,
        "generation_time_ms": elapsed_ms,
        "tokens_generated": usage.completion_tokens if usage else None,
    }


if __name__ == "__main__":
    import json
    # Quick test with dummy chunks
    test_chunks = [
        {
            "content": "Employees may work remotely up to 3 days per week with manager approval.",
            "source": "company_handbook.pdf",
            "page": 4,
            "relevance_score": 0.91,
        }
    ]
    result = generate_answer("How many remote work days are allowed?", test_chunks)
    print(json.dumps(result, indent=2))