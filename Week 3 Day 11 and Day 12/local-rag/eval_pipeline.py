import json
from deepeval.metrics import (
    AnswerRelevancyMetric, FaithfulnessMetric,
    ContextualRelevancyMetric, ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

with open("pipeline_outputs_local.json") as f:
    results = json.load(f)

test_cases = []
for r in results:
    tc = LLMTestCase(
        input=r["question"],
        actual_output=r["actual_answer"],
        expected_output=r["expected_answer"],
        retrieval_context=r["retrieved_context"],
    )
    test_cases.append(tc)

metrics = [
    FaithfulnessMetric(threshold=0.7, include_reason=True),
    AnswerRelevancyMetric(threshold=0.7, include_reason=True),
    ContextualRelevancyMetric(threshold=0.7, include_reason=True),
    ContextualRecallMetric(threshold=0.7, include_reason=True),
    ContextualPrecisionMetric(threshold=0.7, include_reason=True),
]

#eval_results = evaluate(test_cases, metrics) this fails because I get 429 RateLimitError, so ran sequentially
all_results = []
for i, tc in enumerate(test_cases):
    print(f"Evaluating test case {i+1}/{len(test_cases)}: {tc.input[:50]}...")
    result = evaluate([tc], metrics)
    all_results.append(result.model_dump())

with open("eval_results_local.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"Evaluation complete, results saved to eval_results_local.json")