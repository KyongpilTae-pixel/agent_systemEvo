"""BaseAgent — Claude Agent SDK 호환 인터페이스.

Phase 1에서는 Python class 단순 모듈로 동작. Phase 2에서 동일 인터페이스로
SDK Tool wrapper 생성 가능하도록 다음 규약 강제:

  1. 모든 public 메소드는 `tools()` 리스트에 선언 → SDK 도구 자동 등록
  2. 각 메소드는 keyword-only 인자만 받고, `AgentResult` 반환
  3. 인자/반환은 모두 dataclass 또는 JSON-serializable primitive
  4. `describe()`로 SDK가 필요로 하는 메타데이터(이름, 설명, 도구 목록) 제공
  5. 부수 효과(파일 쓰기, state 변경)는 `AgentResult.audit`에 기록 — idempotent
     가 어렵다면 명시적으로 표기
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_system.agents.state_schema import AgentResult, SharedState


@dataclass
class ToolDescription:
    """Claude Agent SDK 도구 정의용 메타데이터."""
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)   # arg → "<type> | <desc>"
    returns: str = "AgentResult"


@dataclass
class AgentDescription:
    """Agent 한 개의 SDK 등록용 메타데이터."""
    code: str               # "A".."H" or "G"
    name: str
    role: str               # one-line description
    tools: list[ToolDescription] = field(default_factory=list)


def tool(description: str):
    """Decorator: agent의 메소드를 SDK 도구로 노출.

    Phase 1: 단순 marker. Phase 2: SDK가 self.tools()에서 발견해 자동 register.
    """
    def deco(fn: Callable) -> Callable:
        fn.__tool_description__ = description
        return fn
    return deco


class BaseAgent(ABC):
    """모든 agent의 공통 인터페이스.

    구현 패턴 예시:

        class MyAgent(BaseAgent):
            CODE = "A"; NAME = "MyAgent"; ROLE = "Does foo bar"

            @tool("Audit available checkpoints")
            def audit(self, *, state: SharedState) -> AgentResult:
                ...
    """
    CODE: str = "?"
    NAME: str = "Agent"
    ROLE: str = ""

    # ---------------- introspection ----------------
    def tools(self) -> list[ToolDescription]:
        """SDK 자동 register용 — @tool 데코레이터 단 메소드 수집."""
        out = []
        for name, fn in inspect.getmembers(self, predicate=inspect.ismethod):
            desc = getattr(fn, "__tool_description__", None)
            if desc is None:
                continue
            sig = inspect.signature(fn)
            params = {pn: f"{p.annotation.__name__ if hasattr(p.annotation, '__name__') else 'Any'}"
                      for pn, p in sig.parameters.items()
                      if pn != "self" and p.kind == inspect.Parameter.KEYWORD_ONLY}
            out.append(ToolDescription(name=name, description=desc, parameters=params))
        return sorted(out, key=lambda t: t.name)

    def describe(self) -> AgentDescription:
        return AgentDescription(
            code=self.CODE, name=self.NAME, role=self.ROLE, tools=self.tools(),
        )

    # ---------------- helpers ----------------
    def _ok(self, method: str, **payload) -> AgentResult:
        return AgentResult(agent=self.CODE, method=method, success=True, **payload)

    def _fail(self, method: str, error: str, **payload) -> AgentResult:
        return AgentResult(agent=self.CODE, method=method, success=False,
                           errors=[error], **payload)
