import json
import time
from openai import OpenAI

# --- Configuration ---
MODELS = ["qwen3:8b", "phi4-mini:latest"]
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"

PROMPTS = [
    {
        "category": "factual",
        "prompt": "What causes tides on Earth? Answer in 2-3 sentences.",
    },
    {
        "category": "reasoning",
        "prompt": (
            "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, "
            "what is the average speed for the entire journey? Show your work."
        ),
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize the following in 3–4 sentences: "
            "Airplanes are fixed-wing aircraft designed for powered flight through the atmosphere. "
            "They generate lift primarily through the movement of air over their wings, which are shaped "
            "to create differences in air pressure. Modern commercial airplanes are typically powered "
            "by jet engines, while smaller aircraft may use propellers. Airplanes are used for passenger "
            "transport, cargo shipment, military operations, scientific research, and firefighting. "
            "The first successful powered flight was achieved by the Wright brothers in 1903, leading "
            "to rapid advancements in aviation technology throughout the 20th and 21st centuries. "
            "Today’s long-haul aircraft can travel thousands of kilometers nonstop and are equipped "
            "with advanced navigation systems, autopilot capabilities, and sophisticated safety mechanisms."
        )
    },
    {
        "category": "structured_output",
        "prompt": (
            'List 3 European capitals with their population (approximate). '
            'Respond ONLY with valid JSON, no extra text: '
            '[{"city": "...", "country": "...", "population_millions": 0.0}]'
        ),
    },
    {
        "category": "code_generation",
        "prompt": (
            "Write a Python function called `chunk_text(text, chunk_size, overlap)` "
            "that splits a string into overlapping chunks. "
            "Include a docstring and a usage example in a comment."
        ),
    },
    {
        "category": "instruction_following",
        "prompt": (
            "List exactly 4 benefits of local LLM deployment over cloud APIs. "
            "Format your response as a numbered list. "
            "Each item must be exactly one sentence. "
            "Do not include any introduction or conclusion."
        ),
    },
    {
        "category": "rag_simulation",
        "prompt": (
            "You MUST answer using ONLY the information provided in the CONTEXT below. "
            "Do not use prior knowledge. If the answer is not explicitly stated in the context, say "
            "'Not found in context.'\n\n"
            "CONTEXT:\n"
            "Lithuania's National Education Framework establishes that students are entitled to free "
            "primary and secondary education in public schools. Students have the right to receive "
            "information about their academic evaluation and to appeal final exam results. "
            "Schools with more than 500 students must appoint a Student Ombudsperson.\n\n"
            "QUESTION:\n"
            "Based only on the CONTEXT above, what rights do students have under Lithuania's National Education Framework? "
            "In your answer, explicitly quote the relevant sentence from the CONTEXT."
        )
    },
]

def run_benchmark(model: str, prompt: str) -> dict:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    usage = response.usage

    return {
        "model": model,
        "response": content,
        "time_seconds": round(elapsed, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "tokens_per_second": round(usage.completion_tokens / elapsed, 1) if usage and usage.completion_tokens else None,
    }
    
# --- Main ---

if __name__ == "__main__":
    all_results = []

    for model in MODELS:
        for p in PROMPTS:
            result = run_benchmark(model, p["prompt"])
            result["category"] = p["category"]
            result["prompt"] = p["prompt"]
            all_results.append(result)

    summary = {}
    for model in MODELS:
        rows = [r for r in all_results if r["model"] == model]
        times = [r["time_seconds"] for r in rows]
        tps   = [r["tokens_per_second"] for r in rows if r["tokens_per_second"]]
        summary[model] = {
            "avg_time_seconds":      round(sum(times) / len(times), 2),
            "avg_tokens_per_second": round(sum(tps) / len(tps), 1) if tps else None,
            "prompts_run":           len(rows),
        }

    output = {
        "metadata": {
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": MODELS,
            "temperature": 0.7,
            "num_prompts": len(PROMPTS),
        },
        "summary": summary,
        "results": all_results,
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)