"""tool-calling 루프. LLM(vLLM OpenAI 호환) ↔ 도구 레지스트리.

한 요청 = 최대 MAX_TOOL_ROUNDS 번 (LLM → tool_calls 실행 → 결과 주입) 후 최종 답.
진행 이벤트를 yield 해서 서버가 스트리밍으로 보여줄 수 있게 한다. 모든 단계를 감사 로그(JSONL)에 남긴다.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Iterator

import requests

from . import config, tools

SYSTEM_PROMPT = (config.HERE / "prompts/system.md").read_text()


def _llm(messages: list[dict], with_tools: bool = True) -> dict:
    body = {"model": config.LLM_MODEL, "messages": messages, "temperature": 0.2,
            "max_tokens": config.LLM_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": config.LLM_THINKING}}
    if with_tools:
        body["tools"] = tools.schemas()
        body["tool_choice"] = "auto"
    r = requests.post(f"{config.LLM_BASE}/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {config.LLM_KEY}"}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _audit(rec: dict):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_DIR / f"audit_{datetime.now():%Y%m%d}.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def run(user_messages: list[dict], user: str = "anonymous") -> Iterator[dict]:
    """이벤트: {"type":"tool","name","args"} · {"type":"tool_result","name","chars"} · {"type":"final","content"} · {"type":"error"}"""
    rid = uuid.uuid4().hex[:12]
    # 클라이언트가 보낸 system 은 버리고 우리 것을 쓴다 (규칙 우회 방지)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m.get("content") or ""} for m in user_messages
        if m.get("role") in ("user", "assistant")]
    _audit({"rid": rid, "t": time.time(), "user": user, "event": "request",
            "question": msgs[-1]["content"] if msgs else ""})
    try:
        for rnd in range(config.MAX_TOOL_ROUNDS):
            m = _llm(msgs)
            calls = m.get("tool_calls") or []
            if not calls:
                content = (m.get("content") or "").strip()
                _audit({"rid": rid, "t": time.time(), "event": "final", "round": rnd, "content": content})
                yield {"type": "final", "content": content}
                return
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments") or "{}"
                yield {"type": "tool", "name": name, "args": args}
                t0 = time.time()
                out = tools.run(name, args)
                _audit({"rid": rid, "t": time.time(), "event": "tool", "round": rnd, "name": name,
                        "args": args, "sec": round(time.time() - t0, 2), "result_head": out[:500]})
                yield {"type": "tool_result", "name": name, "chars": len(out)}
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        # 라운드 소진 → 도구 없이 지금까지로 답하게 한다
        m = _llm(msgs + [{"role": "user", "content": "도구 호출 한도에 도달했다. 지금까지 결과만으로 답하라."}],
                 with_tools=False)
        content = (m.get("content") or "").strip()
        _audit({"rid": rid, "t": time.time(), "event": "final", "round": "limit", "content": content})
        yield {"type": "final", "content": content}
    except Exception as e:  # noqa: BLE001
        _audit({"rid": rid, "t": time.time(), "event": "error", "error": repr(e)})
        yield {"type": "error", "content": f"에이전트 오류: {type(e).__name__}: {e}"}
