"""운영 조회 도구 (읽기 전용): GPU 큐 현황. 큐 편집·등록은 W 등급이라 여기 없음."""
from __future__ import annotations

import re
import subprocess

from .. import config
from . import tool

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@tool("queue_status",
      "통합 GPU 큐 현황(각 GPU 에서 돌고 있는 학습/추론 작업, 경과시간, 대기/중지 항목과 의존관계)을 조회한다. "
      "scripts/gpu_queue_status.sh 출력 그대로. priority 는 숫자가 클수록 먼저 실행된다.",
      {"type": "object",
       "properties": {"section": {"type": "string", "enum": ["summary", "full"],
                                  "description": "summary=GPU 점유·실행중만(기본), full=대기열 포함"}}})
def queue_status(section: str = "summary"):
    r = subprocess.run(["bash", str(config.ROOT / "scripts/gpu_queue_status.sh")],
                       capture_output=True, text=True, timeout=60, cwd=config.ROOT)
    out = _ANSI.sub("", r.stdout)
    if section != "full":
        out = out.split("▸ 통합 실행큐 항목")[0]
    return out.strip() or f"(출력 없음) stderr={r.stderr[-400:]}"
