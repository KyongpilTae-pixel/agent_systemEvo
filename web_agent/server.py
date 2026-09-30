"""Agent 서버 — OpenAI 호환 API 로 노출해서 Open WebUI 가 모델 'drast-agent' 로 붙여 쓴다.

  python -m agent_system.web_agent.server        (cwd = qnt_algorithm)
GET /v1/models · POST /v1/chat/completions (stream true/false) · GET /health
"""
from __future__ import annotations

import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import agent, config, tools

tools.load_all()
app = FastAPI(title="drast-agent")


def _auth(authorization: str | None):
    if authorization != f"Bearer {config.AGENT_KEY}":
        raise HTTPException(401, "bad key")


def _progress(ev: dict) -> str | None:
    if ev["type"] == "tool":
        return f"> 🔧 `{ev['name']}` {ev['args'][:160]}\n"
    return None


@app.get("/health")
def health():
    return {"ok": True, "tools": list(tools.REGISTRY), "llm": config.LLM_MODEL}


@app.get("/v1/models")
def models(authorization: str | None = Header(None)):
    _auth(authorization)
    return {"object": "list", "data": [{"id": config.AGENT_MODEL_ID, "object": "model",
                                        "created": int(time.time()), "owned_by": "qnt"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request, authorization: str | None = Header(None)):
    _auth(authorization)
    body = await req.json()
    user = req.headers.get("x-openwebui-user-email") or body.get("user") or "anonymous"
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time())

    def chunk(text: str | None, finish: str | None = None) -> str:
        d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": config.AGENT_MODEL_ID,
             "choices": [{"index": 0, "delta": ({"content": text} if text else {}), "finish_reason": finish}]}
        return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

    if body.get("stream"):
        def gen():
            for ev in agent.run(body.get("messages", []), user=user):
                p = _progress(ev)
                if p:
                    yield chunk(p)
                elif ev["type"] in ("final", "error"):
                    yield chunk("\n" + ev["content"])
            yield chunk(None, "stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    trace, final = [], ""
    for ev in agent.run(body.get("messages", []), user=user):
        p = _progress(ev)
        if p:
            trace.append(p)
        elif ev["type"] in ("final", "error"):
            final = ev["content"]
    return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": config.AGENT_MODEL_ID,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(trace) + "\n" + final},
                                      "finish_reason": "stop"}]})


if __name__ == "__main__":
    uvicorn.run(app, host=config.AGENT_HOST, port=config.AGENT_PORT)
