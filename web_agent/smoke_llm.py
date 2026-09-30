"""로컬 vLLM 서버 스모크: (1) 한국어 응답 (2) tool call 1회 왕복 (3) 도구 결과 인용 여부.

  python agent_system/web_agent/smoke_llm.py --base http://127.0.0.1:8100/v1 --model qwen3-14b
외부 의존 없음(requests 만). 숫자 환각 체크: 도구가 준 PASS 값(103)을 그대로 인용하는지 본다.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import requests

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_recipe_metrics",
        "description": "운영 recipe 이름으로 FDA2023 임상 PASS 셀 수와 VME 비율을 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {"recipe": {"type": "string", "description": "recipe 이름, 예: routed_iso_t65"}},
            "required": ["recipe"],
        },
    },
}]
FAKE_DB = {"routed_iso_t65": {"pass_cells": 103, "total_cells": 313, "vme_pct": 1.99}}
SYSTEM = "당신은 dRAST 분석 보조 에이전트다. 숫자는 반드시 도구 결과만 인용하고, 없으면 모른다고 답한다. 한국어로 답한다."


def chat(base, key, model, messages, tools=None):
    body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1024}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    t = time.time()
    r = requests.post(f"{base}/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {key}"}, timeout=300)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"], time.time() - t, d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=os.environ.get("LLM_KEY", "local"))
    a = ap.parse_args()
    ok = {}

    m, dt, u = chat(a.base, a.key, a.model, [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": "VME 가 무엇인지 두 문장으로 설명해줘."}])
    print(f"[1] 한국어 ({dt:.1f}s, {u.get('completion_tokens')}tok)\n{m.get('content')}\n")
    ok["korean"] = bool(m.get("content")) and any("가" <= ch <= "힣" for ch in m["content"])

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "운영 권고 recipe routed_iso_t65 의 PASS 셀 수와 VME 알려줘."}]
    m, dt, _ = chat(a.base, a.key, a.model, msgs, TOOLS)
    calls = m.get("tool_calls") or []
    print(f"[2] tool_calls ({dt:.1f}s): {json.dumps(calls, ensure_ascii=False)}")
    ok["tool_call"] = len(calls) == 1 and calls[0]["function"]["name"] == "get_recipe_metrics"
    if ok["tool_call"]:
        args = json.loads(calls[0]["function"]["arguments"] or "{}")
        ok["tool_args"] = args.get("recipe") == "routed_iso_t65"
        msgs += [m, {"role": "tool", "tool_call_id": calls[0]["id"],
                     "content": json.dumps(FAKE_DB.get(args.get("recipe"), {"error": "not found"}))}]
        m, dt, _ = chat(a.base, a.key, a.model, msgs, TOOLS)
        print(f"[3] 최종 ({dt:.1f}s)\n{m.get('content')}\n")
        ok["cites_tool"] = "103" in (m.get("content") or "") and "1.99" in (m.get("content") or "")
    print("RESULT", json.dumps(ok, ensure_ascii=False))
    raise SystemExit(0 if all(ok.values()) and len(ok) == 4 else 1)


if __name__ == "__main__":
    main()
