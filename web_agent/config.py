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
EXTRA_KNOWLEDGE = [Path("/home/kptae/project/drastUtils/README.md")]
