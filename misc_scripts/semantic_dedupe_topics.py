#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rapidfuzz import fuzz
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config
from helper_functions.llm_client import get_client


TOPICS_FILE = Path("helper_functions/lesson_utils.py")


@dataclass(frozen=True)
class TopicItem:
    index: int
    category: str
    text: str


def _parse_topics_with_categories(path: Path) -> list[TopicItem]:
    lines = path.read_text().splitlines()
    in_topics = False
    category = "(uncategorized)"
    items: list[TopicItem] = []

    category_re = re.compile(r"^\s*#\s*-{3,}\s*(.*?)\s*-{3,}\s*$")
    string_re = re.compile(r'^\s*"(.*)"\s*,?\s*$')

    for line in lines:
        if not in_topics:
            if re.match(r"^\s*topics\s*=\s*\[\s*$", line):
                in_topics = True
            continue

        if line.strip() == "]":
            break

        m = category_re.match(line)
        if m:
            category = m.group(1).strip()
            continue

        m = string_re.match(line)
        if m:
            items.append(TopicItem(index=len(items), category=category, text=m.group(1)))

    if not items:
        raise RuntimeError(f"No topics parsed from {path}")
    return items


def _load_topics_via_ast(path: Path) -> list[str]:
    mod = ast.parse(path.read_text())
    for node in mod.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "topics" for t in node.targets
        ):
            topics = ast.literal_eval(node.value)
            if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
                raise RuntimeError("topics is not a list[str]")
            return topics
    raise RuntimeError("Could not find topics = [...] assignment")


def _embedding_model_name() -> tuple[str, bool]:
    model = getattr(config, "litellm_embedding", "")
    model_name = model.split(":", 1)[-1] if ":" in model else model
    is_asymmetric = "nv-embedqa" in model_name or "nv-embed" in model_name
    return model_name, is_asymmetric


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cache[rec["sha"]] = rec["embedding"]
    return cache


def _append_cache(cache_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _get_embeddings_via_litellm(texts: list[str], batch_size: int) -> np.ndarray:
    model_name, is_asymmetric = _embedding_model_name()

    llm = get_client(provider="litellm", model_tier="fast")
    client = llm.client
    extra_body = {"input_type": "passage"} if is_asymmetric else {}

    cache_path = Path("web_search_cache") / f"topic_embeddings_{model_name.replace('/', '_')}.jsonl"
    cache = _load_cache(cache_path)

    embeddings: list[list[float]] = [None] * len(texts)  # type: ignore[assignment]
    missing: list[tuple[int, str, str]] = []

    for i, t in enumerate(texts):
        key = _sha(t)
        if key in cache:
            embeddings[i] = cache[key]
        else:
            missing.append((i, key, t))

    if missing:
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            chunk_texts = [t for _, _, t in chunk]
            resp = client.embeddings.create(
                model=model_name,
                input=chunk_texts,
                extra_body=extra_body,
            )
            new_rows = []
            for (i, key, _), data in zip(chunk, resp.data, strict=False):
                emb = data.embedding
                embeddings[i] = emb
                new_rows.append({"sha": key, "embedding": emb})
            _append_cache(cache_path, new_rows)

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
    return arr


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_kv_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    out: dict[str, Any] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["key"]] = rec["value"]
    return out


def _append_kv_cache(cache_path: Path, key: str, value: Any) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


def _llm_is_duplicate(a: str, b: str) -> dict[str, Any]:
    llm = get_client(provider="litellm", model_tier="fast")
    prompt = (
        "You are deduplicating a list of lesson topics.\n"
        "Decide whether Topic A and Topic B are semantically duplicates (i.e., they would produce essentially the same lesson).\n"
        "Be conservative: if they have meaningfully different emphasis, scope, or examples, they are NOT duplicates.\n\n"
        f"Topic A: {a}\n"
        f"Topic B: {b}\n\n"
        "Return ONLY valid JSON with this schema:\n"
        '{\"duplicate\": true|false, \"keep\": \"A\"|\"B\"|null, \"reason\": \"...\"}\n'
        "If duplicate is false, keep must be null.\n"
        "If duplicate is true, choose which to keep based on clarity and general usefulness."
    )
    raw = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=220,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned non-JSON: {raw[:400]}") from e
    if not isinstance(data, dict) or "duplicate" not in data:
        raise RuntimeError(f"LLM returned unexpected JSON: {raw[:400]}")
    return data


def _llm_is_duplicate_cached(
    llm,  # UnifiedLLMClient
    cache_path: Path,
    cache: dict[str, Any],
    a: str,
    b: str,
) -> dict[str, Any]:
    key = _sha(f"{llm.model}|{a}||{b}")
    if key in cache:
        return cache[key]

    prompt = (
        "You are deduplicating a list of lesson topics.\n"
        "Decide whether Topic A and Topic B are semantically duplicates (i.e., they would produce essentially the same lesson).\n"
        "Be conservative: if they have meaningfully different emphasis, scope, or examples, they are NOT duplicates.\n\n"
        f"Topic A: {a}\n"
        f"Topic B: {b}\n\n"
        "Return ONLY valid JSON with this schema:\n"
        '{\"duplicate\": true|false, \"keep\": \"A\"|\"B\"|null, \"reason\": \"...\"}\n'
        "If duplicate is false, keep must be null.\n"
        "If duplicate is true, choose which to keep based on clarity and general usefulness."
    )
    raw = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=220,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned non-JSON: {raw[:400]}") from e
    if not isinstance(data, dict) or "duplicate" not in data:
        raise RuntimeError(f"LLM returned unexpected JSON: {raw[:400]}")

    _append_kv_cache(cache_path, key, data)
    cache[key] = data
    return data


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sim", type=float, default=0.93, help="Cosine similarity threshold")
    ap.add_argument("--neighbors", type=int, default=8, help="Neighbors to check per topic")
    ap.add_argument("--min-fuzz", type=float, default=0.0, help="Optional RapidFuzz token_set_ratio gate (0-100)")
    ap.add_argument("--within-category", action="store_true", help="Only compare topics within the same category")
    ap.add_argument("--max-pairs", type=int, default=250, help="Max candidate pairs to send to the LLM")
    ap.add_argument("--batch-size", type=int, default=int(getattr(config, "embed_batch_size", 10)))
    ap.add_argument("--report", type=Path, default=Path("web_search_cache/topic_dedupe_report.json"))
    ap.add_argument("--apply", action="store_true", help="Write deduped topics back to helper_functions/lesson_utils.py")
    args = ap.parse_args()

    items = _parse_topics_with_categories(TOPICS_FILE)
    topics = [it.text for it in items]
    assert topics == _load_topics_via_ast(TOPICS_FILE), "Text parser and AST parser disagree"

    emb = _get_embeddings_via_litellm(topics, batch_size=args.batch_size)
    emb = _normalize_rows(emb)

    nn = NearestNeighbors(n_neighbors=min(args.neighbors + 1, len(topics)), metric="cosine", algorithm="brute")
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb)

    llm = get_client(provider="litellm", model_tier="fast")
    llm_decision_cache = Path("web_search_cache") / "topic_dedupe_decisions.jsonl"
    llm_decisions = _load_kv_cache(llm_decision_cache)

    candidate_pairs: list[tuple[int, int, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(topics)):
        for dist, j in zip(distances[i][1:], indices[i][1:], strict=False):
            sim = 1.0 - float(dist)
            if sim < args.min_sim:
                continue
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if (a, b) in seen:
                continue
            if args.within_category and items[a].category != items[b].category:
                continue
            seen.add((a, b))
            fuzz_score = float(fuzz.token_set_ratio(topics[a], topics[b]))
            if fuzz_score < args.min_fuzz:
                continue
            candidate_pairs.append((a, b, sim, fuzz_score))

    candidate_pairs.sort(key=lambda x: (-x[2], -x[3]))
    truncated = False
    if args.max_pairs and len(candidate_pairs) > args.max_pairs:
        candidate_pairs = candidate_pairs[: args.max_pairs]
        truncated = True

    decisions: list[dict[str, Any]] = []
    uf = UnionFind(len(topics))
    keep_votes = [0] * len(topics)

    for k, (a, b, sim, fuzz_score) in enumerate(candidate_pairs, start=1):
        if k % 25 == 0 or k == 1 or k == len(candidate_pairs):
            print(f"Checked {k}/{len(candidate_pairs)} candidate pairs...")
        res = _llm_is_duplicate_cached(llm, llm_decision_cache, llm_decisions, topics[a], topics[b])
        dup = bool(res.get("duplicate", False))
        keep = res.get("keep", None)
        record = {
            "a": a,
            "b": b,
            "sim": sim,
            "fuzz": fuzz_score,
            "duplicate": dup,
            "keep": keep,
            "reason": res.get("reason", ""),
            "topic_a": topics[a],
            "topic_b": topics[b],
            "category_a": items[a].category,
            "category_b": items[b].category,
        }
        decisions.append(record)
        if not dup:
            continue
        uf.union(a, b)
        if keep == "A":
            keep_votes[a] += 1
        elif keep == "B":
            keep_votes[b] += 1

    # Choose representative per component
    comps: dict[int, list[int]] = {}
    for i in range(len(topics)):
        comps.setdefault(uf.find(i), []).append(i)

    representatives: dict[int, int] = {}
    drops: set[int] = set()
    for root, members in comps.items():
        if len(members) == 1:
            representatives[root] = members[0]
            continue
        members_sorted = sorted(members, key=lambda idx: (-keep_votes[idx], idx))
        rep = members_sorted[0]
        representatives[root] = rep
        for m in members:
            if m != rep:
                drops.add(m)

    report = {
        "source_file": str(TOPICS_FILE),
        "original_count": len(topics),
        "candidates_checked": len(candidate_pairs),
        "candidates_truncated": truncated,
        "duplicates_removed": len(drops),
        "new_count": len(topics) - len(drops),
        "min_sim": args.min_sim,
        "min_fuzz": args.min_fuzz,
        "within_category": args.within_category,
        "max_pairs": args.max_pairs,
        "decisions": decisions,
        "drops": [
            {
                "index": i,
                "category": items[i].category,
                "topic": topics[i],
                "kept_index": representatives[uf.find(i)],
                "kept_topic": topics[representatives[uf.find(i)]],
            }
            for i in sorted(drops)
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report to {args.report}")
    print(f"Original topics: {len(topics)}")
    print(f"Candidates checked: {len(candidate_pairs)}")
    print(f"Duplicates removed: {len(drops)}")
    print(f"New topics: {len(topics) - len(drops)}")

    if not args.apply:
        return 0

    # Apply by deleting exact string lines (conservative, preserves formatting otherwise)
    drop_texts = {topics[i] for i in drops}
    src_lines = TOPICS_FILE.read_text(encoding="utf-8").splitlines(keepends=True)

    def is_topic_line(line: str) -> bool:
        return bool(re.match(r'^\s*\".*\"\,?\s*$', line))

    out_lines: list[str] = []
    removed = 0
    for line in src_lines:
        if is_topic_line(line):
            m = re.match(r'^\s*\"(.*)\"\,?\s*$', line)
            if m and m.group(1) in drop_texts:
                removed += 1
                continue
        out_lines.append(line)

    if removed != len(drops):
        raise RuntimeError(f"Planned to remove {len(drops)} topics but removed {removed} lines; aborting.")

    TOPICS_FILE.write_text("".join(out_lines), encoding="utf-8")
    print(f"Applied: removed {removed} topics from {TOPICS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
