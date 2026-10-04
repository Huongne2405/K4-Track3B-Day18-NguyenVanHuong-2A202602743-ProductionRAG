from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


@lru_cache(maxsize=1)
def _semantic_model():
    """Load một lần, dùng lại model cho nhiều documents."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    metadata = metadata or {}
    sentences = [sentence.strip() for sentence in
                 re.split(r"(?<=[.!?])\s+|\n\n", text) if sentence.strip()]
    if not sentences:
        return []

    groups = [[sentences[0]]]
    if len(sentences) > 1:
        import numpy as np

        embeddings = np.asarray(_semantic_model().encode(sentences))
        for i, sentence in enumerate(sentences[1:], start=1):
            previous, current = embeddings[i - 1], embeddings[i]
            denominator = np.linalg.norm(previous) * np.linalg.norm(current)
            similarity = float(np.dot(previous, current) / max(denominator, 1e-9))
            if similarity < threshold:
                groups.append([])
            groups[-1].append(sentence)

    return [Chunk(text=" ".join(group), metadata={**metadata,
                  "strategy": "semantic", "chunk_index": i})
            for i, group in enumerate(groups)]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_bounded(text: str, size: int) -> list[str]:
    """Ưu tiên paragraph/newline/word boundaries, kể cả paragraph quá dài."""
    pieces = []
    remaining = text.strip()
    while len(remaining) > size:
        split_at = size
        for separator in ("\n\n", "\n", " "):
            boundary = remaining.rfind(separator, 0, size + 1)
            if boundary > 0:
                split_at = boundary
                break
        piece = remaining[:split_at].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent,
        lưu ở cả thuộc tính parent_id và metadata["parent_id"].
    """
    if not 0 < child_size < parent_size:
        raise ValueError("sizes must satisfy 0 < child_size < parent_size")
    metadata = metadata or {}
    # Namespace theo document để parent_0 của hai documents không trùng nhau.
    document_key = sha256((str(metadata.get("source", "")) + "\0" + text)
                          .encode("utf-8")).hexdigest()[:16]
    parents, children = [], []
    for i, parent_text in enumerate(_split_bounded(text, parent_size)):
        pid = f"parent_{document_key}_{i}"
        parents.append(Chunk(text=parent_text, metadata={**metadata,
                             "strategy": "hierarchical", "chunk_type": "parent",
                             "chunk_index": i, "parent_id": pid}))
        for child_text in _split_bounded(parent_text, child_size):
            children.append(Chunk(text=child_text, metadata={**metadata,
                                  "strategy": "hierarchical", "chunk_type": "child",
                                  "chunk_index": len(children), "parent_id": pid},
                                  parent_id=pid))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks = []
    section = ""
    lines = []
    fence = ""

    def flush():
        content = "".join(lines).strip()
        if content:
            chunks.append(Chunk(text=content, metadata={**metadata,
                                "section": section, "strategy": "structure",
                                "chunk_index": len(chunks)}))
        lines.clear()

    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            lines.append(line)
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence)
                    and not line[marker.end():].strip()):
                fence = ""
            continue
        if marker:
            fence = marker[1]
        elif re.match(r"^ {0,3}#{1,6}\s+\S", line):
            flush()
            section = line.strip()
        lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
