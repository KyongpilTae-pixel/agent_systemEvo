"""지식 도구: glossary(용어 정본) · search_knowledge(문서 BM25 검색).

임베딩(bge-m3) 하이브리드는 다음 단계. 지금은 순수 파이썬 BM25 — 원천 ~2MB 라 기동 시 전량 색인.
한국어는 형태소기 없이 음절 bigram + 영숫자 토큰으로 색인한다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .. import config
from . import tool

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*")
_HANGUL = re.compile(r"[가-힣]+")


def _tokens(text: str) -> list[str]:
    t = text.lower()
    toks = _WORD.findall(t)
    for run in _HANGUL.findall(t):
        toks += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


def _chunks(path: Path) -> list[dict]:
    """마크다운을 제목 단위로 자르고, 긴 절은 ~1500자로 다시 자른다."""
    text = path.read_text(errors="ignore")
    rel = str(path.relative_to(config.ROOT)) if path.is_relative_to(config.ROOT) else str(path)
    out, head, buf = [], path.stem, []

    def flush():
        body = "\n".join(buf).strip()
        for i in range(0, len(body), 1500):
            if body[i:i + 1500].strip():
                out.append({"source": rel, "section": head, "text": body[i:i + 1500]})

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            head, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    flush()
    return out


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(_tokens(d["section"] + " " + d["text"])) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[float, dict]]:
        q = set(_tokens(query))
        scored = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((s, self.docs[i]))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


@lru_cache(maxsize=1)
def _index() -> BM25:
    files: dict[Path, None] = {}
    for g in config.KNOWLEDGE_GLOBS:
        for p in sorted(config.ROOT.glob(g)):
            files[p] = None
    for p in config.EXTRA_KNOWLEDGE:
        if p.exists():
            files[p] = None
    docs = [c for p in files for c in _chunks(p)]
    return BM25(docs)


def reindex() -> int:
    _index.cache_clear()
    return len(_index().docs)


@tool("search_knowledge",
      "사내 dRAST 문서(용어사전·일일보고·설계문서·README)를 키워드로 검색해 관련 구절과 출처 파일을 돌려준다. "
      "과거 결정·실험 결과·규약을 물으면 먼저 이것을 쓴다.",
      {"type": "object",
       "properties": {"query": {"type": "string", "description": "검색어 (한국어/영어, 약어 그대로 가능)"},
                      "k": {"type": "integer", "description": "결과 수(기본 5, 최대 8)"}},
       "required": ["query"]})
def search_knowledge(query: str, k: int = 5):
    hits = _index().search(query, min(max(int(k), 1), 8))
    if not hits:
        return {"results": [], "note": "일치 문서 없음 — 모른다고 답할 것"}
    return {"지시": "답변에 참고한 source 파일 경로를 반드시 적을 것",
            "results": [{"source": d["source"], "section": d["section"], "score": round(s, 2),
                         "text": d["text"][:900]} for s, d in hits]}


def _glossary_rows() -> list[tuple[str, str]]:
    """GLOSSARY.md 의 표 행과 불릿을 (키, 전체줄) 로 만든다."""
    rows = []
    for line in (config.ROOT / "claudeCode/GLOSSARY.md").read_text().splitlines():
        s = line.strip()
        if s.startswith("|") and not set(s) <= set("|-: "):
            first = s.strip("|").split("|")[0]
            rows.append((re.sub(r"[*`~]", "", first).strip().lower(), s))
        elif s.startswith("- **"):
            m = re.match(r"- \*\*(.+?)\*\*", s)
            if m:
                rows.append((m.group(1).lower(), s))
    return rows


@tool("glossary",
      "dRAST 용어 정본(claudeCode/GLOSSARY.md)에서 용어 정의를 찾는다. VME·ME·EA·CA·PASS·same_mic·dynamic 등 "
      "약어·데이터셋·모델 이름의 뜻을 물으면 반드시 이것부터 쓴다 (일반 IT 용어로 해석 금지).",
      {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]})
def glossary(term: str):
    t = term.strip().lower()
    rows = _glossary_rows()
    exact = [r for k, r in rows if k == t or re.split(r"[\s(/]", k)[0] == t]
    part = [r for k, r in rows if t in k and r not in exact]
    body = [r for k, r in rows if t in r.lower() and r not in exact and r not in part]
    found = (exact + part + body)[:8]
    if not found:
        return {"term": term, "found": False, "note": "용어사전에 없음. search_knowledge 로 재검색하거나 모른다고 답할 것"}
    return {"term": term, "found": True, "source": "claudeCode/GLOSSARY.md", "entries": found}
