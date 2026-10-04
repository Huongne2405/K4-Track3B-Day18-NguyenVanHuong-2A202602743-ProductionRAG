from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys
from dataclasses import asdict, dataclass
from math import isfinite

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run 4 RAGAS metrics; fallback có error và không có điểm từng câu giả."""
    fallback = {**dict.fromkeys(METRIC_NAMES, 0.0), "per_question": []}
    try:
        if not len(questions) == len(answers) == len(contexts) == len(ground_truths):
            raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
        if not questions:
            return fallback
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        from unittest.mock import patch

        from datasets import Dataset
        from ragas import evaluate
        from ragas.executor import as_completed
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # RAGAS 0.1 tạo tasks trước asyncio.run; trì hoãn đến khi loop chạy.
        # Wrapper được khôi phục sau evaluate, không sửa global asyncio.
        def lazy_as_completed(*args, **kwargs):
            yield from as_completed(*args, **kwargs)

        with patch("ragas.executor.as_completed", lazy_as_completed):
            result = evaluate(
                dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                raise_exceptions=True,
                run_config=RunConfig(timeout=60, max_retries=2, max_wait=10, max_workers=4),
            )
        df = result.to_pandas()
        if len(df) != len(questions):
            raise ValueError("RAGAS returned an unexpected number of rows")
        per_question = []
        for _, row in df.iterrows():
            scores = {name: float(row[name]) for name in METRIC_NAMES}
            if not all(isfinite(score) for score in scores.values()):
                raise ValueError("RAGAS returned non-finite metric scores")
            per_question.append(EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"], **scores,
            ))
        return {
            **{name: sum(getattr(item, name) for item in per_question) / len(per_question)
               for name in METRIC_NAMES},
            "per_question": per_question,
        }
    except Exception as exc:  # noqa: BLE001 -- RAGAS/API failures must not stop the pipeline.
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        return {**fallback, "error": str(exc)}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu",
            "Thắt chặt system prompt, giảm temperature về 0",
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng",
            "Cải thiện bước cắt đoạn hoặc bổ sung từ khóa BM25",
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu",
            "Bổ sung Cross-Encoder reranking hoặc lọc theo metadata",
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn",
        ),
    }
    failures = []
    for result in eval_results:
        scores = {name: getattr(result, name) for name in METRIC_NAMES}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question, "worst_metric": worst_metric,
            "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        })
    return sorted(failures, key=lambda item: item["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save aggregate scores, per-question evidence and failures to JSON."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {name: results.get(name, 0.0) for name in METRIC_NAMES},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [asdict(item) for item in results.get("per_question", [])],
        "failures": failures,
    }
    if "error" in results:
        report["error"] = results["error"]
    if "latency" in results:
        report["latency"] = results["latency"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
