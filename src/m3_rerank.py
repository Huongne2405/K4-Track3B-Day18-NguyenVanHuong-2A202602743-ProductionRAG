from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass
from numbers import Real

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, Real):
            scores = [scores]
        scored = sorted(zip(scores, documents, strict=True), key=lambda item: item[0], reverse=True)
        return [RerankResult(text=doc["text"], original_score=float(doc.get("score", 0.0)),
                             rerank_score=float(score), metadata=dict(doc.get("metadata", {})), rank=i)
                for i, (score, doc) in enumerate(scored[:top_k])]


class FlashrankReranker:
    """Optional ONNX reranker; benchmark latency on the target hardware."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        from flashrank import Ranker, RerankRequest

        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
        ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
        ranked = sorted(ranked, key=lambda item: float(item["score"]), reverse=True)
        results = []
        for rank, passage in enumerate(ranked[:top_k]):
            doc = documents[passage["id"]]
            results.append(RerankResult(
                text=doc["text"], original_score=float(doc.get("score", 0.0)),
                rerank_score=float(passage["score"]),
                metadata=dict(doc.get("metadata", {})), rank=rank,
            ))
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
