"""도구 레지스트리. 도구 = 정본 코드/산출물의 얇은 래퍼 + JSON schema + 권한 등급.

등급: R(읽기, 부작용 없음) · C(CPU 계산, agent 전용 출력 디렉터리에만 씀) · W(공용 자원 변경, 승인 필요 — 미구현)
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .. import config


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    grade: str = "R"

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict, grade: str = "R"):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, parameters, fn, grade)
        return fn
    return deco


def schemas() -> list[dict]:
    return [t.schema() for t in REGISTRY.values() if t.grade in ("R", "C")]


def run(name: str, args_json: str) -> str:
    """도구 실행 → LLM 에 넣을 문자열. 실패도 문자열로 돌려 LLM 이 사용자에게 설명하게 한다."""
    t = REGISTRY.get(name)
    if t is None:
        return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
    try:
        args = json.loads(args_json or "{}")
        out = t.fn(**args)
    except TypeError as e:
        return json.dumps({"error": f"bad arguments: {e}"}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 — 도구 오류는 사용자에게 그대로 보고
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc(limit=2)[-600:]}, ensure_ascii=False)
    s = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
    if len(s) > config.TOOL_RESULT_MAX_CHARS:
        s = s[: config.TOOL_RESULT_MAX_CHARS] + f"\n...[truncated {len(s) - config.TOOL_RESULT_MAX_CHARS} chars]"
    return s


def load_all():
    import importlib
    for m in config.TOOL_MODULES:   # 등록 부작용. 배포 대상별로 켤 도구 묶음을 고른다
        importlib.import_module(f"{__name__}.{m}")
