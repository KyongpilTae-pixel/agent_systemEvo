"""web_agent 설정. 환경변수로 덮어쓴다 (운영 서버 이전 시 경로만 바꾸면 되도록)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("QNT_ROOT", "/home/kptae/project/qnt_algorithm"))
HERE = Path(__file__).resolve().parent

# LLM (vLLM OpenAI 호환) — serve_llm.sh 와 짝
LLM_BASE = os.environ.get("LLM_BASE", "http://127.0.0.1:8100/v1")
LLM_KEY = os.environ.get("LLM_KEY", "local")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")   # 2026-09-30 bake-off 1위
LLM_THINKING = os.environ.get("LLM_THINKING", "0") == "1"   # Qwen3 thinking. 컨텍스트 12k 라 기본 off
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

# Agent 서버 — Open WebUI 가 이 주소를 OpenAI 호환 백엔드로 붙인다
AGENT_HOST = os.environ.get("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.environ.get("AGENT_PORT", "8200"))
AGENT_KEY = os.environ.get("AGENT_KEY", "agent-local")
AGENT_MODEL_ID = "drast-agent"
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "8"))
TOOL_RESULT_MAX_CHARS = int(os.environ.get("TOOL_RESULT_MAX_CHARS", "6000"))  # 컨텍스트 보호

LOG_DIR = Path(os.environ.get("AGENT_LOG_DIR", HERE / "logs"))

# 지식 원천 (search_knowledge). 개인 메모리(~/.claude)는 팀 공유 대상이 아니므로 제외.
KNOWLEDGE_GLOBS = [
    "claudeCode/GLOSSARY.md",
    "claudeCode/RESUME.md",
    "claudeCode/*.md",
    "claudeCode/daily_reports/*.md",
    "claudeCode/total_summary/INDEX.md",
    "agent_system/*.md",
]
EXTRA_KNOWLEDGE = [Path(p) for p in os.environ.get(
    "EXTRA_KNOWLEDGE", "/home/kptae/project/drastUtils/README.md").split(os.pathsep) if p]

# 평가 산출물 위치 — 운영 PC(Windows)로 옮길 때 이 환경변수만 바꾼다. 없는 경로의 도구는 "자료 없음" 오류를 돌려준다.
R25_NEWMODEL = Path(os.environ.get("R25_NEWMODEL", "/home/kptae/project/drast_25_lrcn/newmodel"))
F30_DEPLOY_REPRO = Path(os.environ.get(
    "F30_DEPLOY_REPRO", "/home/kptae/data/allinfo/analytical/FDA_Analytical_Reproducibility.xlsx"))
PY25 = os.environ.get("PY25", "/home/kptae/miniconda3/envs/qnt_algorithm/bin/python")   # 2.5 셀 계산용(lightgbm 필요)
# 팀 전용 도구 묶음: 쉼표 목록. 다른 팀 배포에서는 "knowledge" 만 켠다.
TOOL_MODULES = [m for m in os.environ.get("TOOL_MODULES", "knowledge,ops,metrics").split(",") if m]
