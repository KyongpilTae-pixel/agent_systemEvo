"""골든셋 채점 — 현재 떠 있는 LLM(LLM_MODEL)으로 agent.run 을 돌려 도구선택·필수인용·금지표현을 자동 채점.

  LLM_MODEL=qwen3-14b python -m agent_system.web_agent.eval.run_golden [--tag qwen3-14b] [--only g07,g11]
결과: eval/results/<tag>.json. must 항목의 리스트는 "그 중 하나만 있으면 통과"(OR).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .. import agent, config, tools

HERE = Path(__file__).resolve().parent


def _has(text: str, item) -> bool:
    return any(x in text for x in item) if isinstance(item, list) else item in text


def score(case: dict, called: list[str], final: str) -> dict:
    tool_ok = all(any(c in called for c in (t if isinstance(t, list) else [t])) for t in case["tools"])
    miss = [m for m in case["must"] if not _has(final, m)]
    bad = [f for f in case.get("forbid", []) if f in final]
    bad += [r for r in case.get("forbid_re", []) if re.search(r, final)]
    return {"tool_ok": tool_ok, "must_ok": not miss, "forbid_ok": not bad, "missing": miss, "forbidden": bad,
            "pass": tool_ok and not miss and not bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=config.LLM_MODEL)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    tools.load_all()
    cases = json.loads((HERE / "golden.json").read_text())
    if a.only:
        cases = [c for c in cases if c["id"] in a.only.split(",")]
    out = []
    for c in cases:
        t0, called, final = time.time(), [], ""
        for ev in agent.run([{"role": "user", "content": c["q"]}], user=f"golden:{a.tag}"):
            if ev["type"] == "tool":
                called.append(ev["name"])
            elif ev["type"] in ("final", "error"):
                final = ev["content"]
        s = score(c, called, final)
        out.append({"id": c["id"], "type": c["type"], "q": c["q"], "called": called, "sec": round(time.time() - t0, 1),
                    **s, "answer": final})
        print(f"{c['id']} {'PASS' if s['pass'] else 'FAIL'} {c['type']:4s} {out[-1]['sec']:5.1f}s tools={called} "
              f"miss={s['missing']} bad={s['forbidden']}", flush=True)
    n = len(out)
    summ = {"tag": a.tag, "n": n, "pass": sum(r["pass"] for r in out), "tool_ok": sum(r["tool_ok"] for r in out),
            "must_ok": sum(r["must_ok"] for r in out), "forbid_ok": sum(r["forbid_ok"] for r in out),
            "mean_sec": round(sum(r["sec"] for r in out) / max(n, 1), 1)}
    print("SUMMARY", json.dumps(summ, ensure_ascii=False))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.tag}.json").write_text(json.dumps({"summary": summ, "cases": out}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
