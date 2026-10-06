"""운영 조회 도구 (읽기 전용): GPU 큐 현황. 큐 편집·등록은 W 등급이라 여기 없음."""
from __future__ import annotations

import re
import shutil
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
    script, snap = config.ROOT / "scripts/gpu_queue_status.sh", config.HERE / "exports/queue_status.txt"
    live = script.exists() and shutil.which("bash") and shutil.which("nvidia-smi")
    if live:
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60, cwd=config.ROOT)
        out, head = _ANSI.sub("", r.stdout), ""
    elif snap.exists():   # 운영 PC: 학습 서버가 내보낸 스냅샷을 읽는다
        import time
        out = _ANSI.sub("", snap.read_text(errors="ignore"))
        head = f"[스냅샷 {time.strftime('%Y-%m-%d %H:%M', time.localtime(snap.stat().st_mtime))} 기준 — 실시간 아님]\n"
    else:
        return {"error": "GPU 큐 현황 자료 없음(학습 서버 스냅샷 미생성)"}
    if section != "full":
        out = out.split("▸ 통합 실행큐 항목")[0]
    return head + out.strip() or "(출력 없음)"
