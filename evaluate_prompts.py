import os
import time
import json
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
from difflib import SequenceMatcher

import openai

# -----------------------------
# 0. Config
# -----------------------------
MODEL_UNDER_TEST = "gpt-4o-mini"   # or "gpt-4o"
JUDGE_MODEL = "gpt-4o-mini"

# Thresholds for CI to pass
MIN_CLASSIFICATION_ACCURACY = 0.65    # 65%
MIN_AVG_SIMILARITY = 0.6              # 0–1
MIN_AVG_JUDGE_SCORE = 3.5             # 1–5

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

openai.api_key = os.environ["OPENAI_API_KEY"]

# -----------------------------
# 1. Test Dataset
# -----------------------------
# Small demo dataset: 3 summarization, 3 classification, 3 reasoning
EVAL_DATA: List[Dict[str, Any]] = [
    # Summarization tasks
    {
        "id": "sum1",
        "task_type": "summarization",
        "input": (
            "Cloud computing lets organizations rent computing resources over the internet "
            "instead of buying servers. It offers flexibility and scalability, but also "
            "requires careful security and cost management."
        ),
        "gold": "Cloud computing means renting computing resources online, bringing flexibility and scale but requiring strong security and cost control."
    },
    {
        "id": "sum2",
        "task_type": "summarization",
        "input": (
            "Containerization packages applications and their dependencies into isolated units "
            "called containers. This improves portability and consistency across environments."
        ),
        "gold": "Containerization bundles apps and dependencies into containers, making them portable and consistent across environments."
    },
    # Classification tasks
    {
        "id": "cls1",
        "task_type": "classification",
        "input": "The service has been down for 2 hours and customers are angry.",
        "label": "negative",
    },
    {
        "id": "cls2",
        "task_type": "classification",
        "input": "The deployment completed successfully and performance improved.",
        "label": "positive",
    },
    {
        "id": "cls3",
        "task_type": "classification",
        "input": "The system is running as expected with no major changes.",
        "label": "neutral",
    },
    # Reasoning tasks
    {
        "id": "reason1",
        "task_type": "reasoning",
        "input": "A server handles 120 requests per minute. How many requests in 10 minutes?",
        "gold": "1200",
    },
    {
        "id": "reason2",
        "task_type": "reasoning",
        "input": "If a backup runs every 4 hours, how many times does it run in 24 hours?",
        "gold": "6",
    },
]

# -----------------------------
# 2. Prompt Template Under Test
# -----------------------------
def build_prompt(example: Dict[str, Any]) -> str:
    """
    Single prompt template that handles all three task types.
    This is what you're 'benchmarking' in CI.
    """
    if example["task_type"] == "summarization":
        return (
            "You are a helpful assistant.\n\n"
            "Task: Summarize the following text in 1–2 clear sentences.\n\n"
            f"Text:\n'''{example['input']}'''"
        )
    if example["task_type"] == "classification":
        return (
            "Classify the sentiment of the following text as one of: positive, negative, or neutral.\n\n"
            f"Text:\n'''{example['input']}'''\n\n"
            "Answer with a single word: positive, negative, or neutral."
        )
    if example["task_type"] == "reasoning":
        return (
            "Solve the following problem. Answer with a single number only.\n\n"
            f"Problem:\n'''{example['input']}'''"
        )
    raise ValueError(f"Unknown task_type: {example['task_type']}")

# -----------------------------
# 3. LLM Call Helpers
# -----------------------------
def call_llm(prompt: str, model: str = MODEL_UNDER_TEST, max_tokens: int = 250) -> str:
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def call_judge(prompt: str, model: str = JUDGE_MODEL, max_tokens: int = 350) -> str:
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# 4. Simple Similarity Metric
# -----------------------------
def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# -----------------------------
# 5. LLM-as-a-Judge for open tasks
# -----------------------------
def judge_prediction(example: Dict[str, Any], prediction: str) -> float:
    """
    Ask a judge model to rate how good the prediction is vs gold (1–5).
    Used for summarization and reasoning.
    """
    if "gold" not in example:
        return 0.0
    judge_prompt = f"""You are an impartial evaluator.

Task: Rate how well the model's answer matches the reference answer for the user task.

User task type: {example['task_type']}

User input:
'''{example['input']}'''

Reference (ideal) answer:
'''{example['gold']}'''

Model's answer:
'''{prediction}'''

Score the model's answer from 1 to 5, where:
1 = very poor or wrong,
2 = partially correct but with big issues,
3 = somewhat correct but missing important parts,
4 = mostly correct with minor issues,
5 = excellent and matches the reference closely.

Respond with ONLY a single number between 1 and 5.
"""
    raw = call_judge(judge_prompt)
    try:
        score = float(raw.strip())
    except ValueError:
        score = 3.0
    return max(1.0, min(5.0, score))

# -----------------------------
# 6. Evaluation Loop
# -----------------------------
@dataclass
class Metrics:
    classification_correct: int = 0
    classification_total: int = 0
    similarity_sum: float = 0.0
    similarity_count: int = 0
    judge_sum: float = 0.0
    judge_count: int = 0

def evaluate_prompt_on_dataset(data: List[Dict[str, Any]]) -> Metrics:
    metrics = Metrics()
    for ex in data:
        prompt = build_prompt(ex)
        print(f" Evaluating example {ex['id']} ({ex['task_type']})...")
        start = time.time()
        prediction = call_llm(prompt)
        elapsed = time.time() - start
        print(f"   Model output: {prediction!r} (time: {elapsed:.2f}s)")
        if ex["task_type"] == "classification":
            gold = ex["label"].strip().lower()
            pred_norm = prediction.strip().lower()
            if gold in pred_norm:
                metrics.classification_correct += 1
            metrics.classification_total += 1
        if ex["task_type"] in ("summarization", "reasoning"):
            gold = ex["gold"]
            sim = string_similarity(prediction, gold)
            metrics.similarity_sum += sim
            metrics.similarity_count += 1
            print(f"   Similarity vs gold: {sim:.2f}")
            judge_score = judge_prediction(ex, prediction)
            metrics.judge_sum += judge_score
            metrics.judge_count += 1
            print(f"   Judge score: {judge_score:.1f}")
        print()
    return metrics

# -----------------------------
# 7. Aggregate + CI Gate
# -----------------------------
def main():
    print(" Running prompt benchmark on test dataset...\n")
    metrics = evaluate_prompt_on_dataset(EVAL_DATA)
    acc = (
        metrics.classification_correct / metrics.classification_total
        if metrics.classification_total
        else 0.0
    )
    avg_sim = (
        metrics.similarity_sum / metrics.similarity_count
        if metrics.similarity_count
        else 0.0
    )
    avg_judge = (
        metrics.judge_sum / metrics.judge_count
        if metrics.judge_count
        else 0.0
    )
    print(" Aggregate Metrics")
    print(f"- Classification accuracy : {acc:.2f}")
    print(f"- Avg similarity (0–1)    : {avg_sim:.2f}")
    print(f"- Avg judge score (1–5)   : {avg_judge:.2f}\n")
    passed = True
    if acc < MIN_CLASSIFICATION_ACCURACY:
        print(f" FAIL: accuracy {acc:.2f} < threshold {MIN_CLASSIFICATION_ACCURACY:.2f}")
        passed = False
    if avg_sim < MIN_AVG_SIMILARITY:
        print(f" FAIL: avg similarity {avg_sim:.2f} < threshold {MIN_AVG_SIMILARITY:.2f}")
        passed = False
    if avg_judge < MIN_AVG_JUDGE_SCORE:
        print(f" FAIL: avg judge score {avg_judge:.2f} < threshold {MIN_AVG_JUDGE_SCORE:.2f}")
        passed = False
    if passed:
        print(" All thresholds met. Prompt template PASSES the benchmark.")
        sys.exit(0)
    else:
        print(" Thresholds not met. Failing CI build.")
        sys.exit(1)

if __name__ == "__main__":
    main()
