#!/usr/bin/env bash
# 테스트 스택 일괄 기동/종료: vLLM(serve_llm.sh) → Agent 서버(8200) → Open WebUI(8300).
#   serve_stack.sh start | stop | status
# 전부 127.0.0.1 바인딩. 브라우저는 VS Code 포트 포워딩(8300)으로 접속.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUN=$HOME/llm_models/_run; mkdir -p "$RUN" "$HOME/llm_models/_webui"
W=$HOME/miniconda3/envs/openwebui
SECRET_FILE=$HOME/llm_models/_webui/.secret
[ -f "$SECRET_FILE" ] || (umask 077; openssl rand -hex 32 > "$SECRET_FILE")

stop_pg(){ local f=$RUN/$1.pid; [ -f "$f" ] || return 0; local p; p=$(cat "$f")
  kill -TERM -- "-$p" 2>/dev/null; for _ in $(seq 1 20); do pgrep -g "$p" >/dev/null || break; sleep 1; done
  pgrep -g "$p" >/dev/null && kill -KILL -- "-$p"; rm -f "$f"; echo "stopped $1"; }

wait_http(){ for _ in $(seq 1 "$3"); do curl -sf "$1" >/dev/null && { echo "READY $2"; return 0; }
  kill -0 "$(cat "$RUN/$2.pid")" 2>/dev/null || { echo "DIED $2 — $RUN/$2.log"; return 1; }; sleep 3; done; echo "timeout $2"; return 1; }

case "${1:-status}" in
start)
  "$HERE/serve_llm.sh" status >/dev/null 2>&1 && curl -sf 127.0.0.1:8100/v1/models -H "Authorization: Bearer ${LLM_KEY:-local}" >/dev/null \
    || "$HERE/serve_llm.sh" start auto "${LLM_MODEL:-qwen3.8-27b}" || exit 1
  # ★'cd X && cmd &' 는 서브셸이 백그라운드되어 $! 가 실제 프로세스가 아님 → cd 를 먼저, 명령만 백그라운드
  for port in 8200 8300; do curl -s 127.0.0.1:$port >/dev/null 2>&1 && { echo "port $port 사용 중 — 잔존 프로세스 확인: pgrep -af 'web_agent.server|run_webui'"; exit 4; }; done
  cd "$ROOT"
  LLM_MODEL=${LLM_MODEL:-qwen3.8-27b} setsid nohup python -m agent_system.web_agent.server > "$RUN/agent.log" 2>&1 &
  echo $! > "$RUN/agent.pid"; wait_http 127.0.0.1:8200/health agent 20 || exit 1
  cd "$HOME/llm_models/_webui"
  DATA_DIR=$HOME/llm_models/_webui WEBUI_SECRET_KEY=$(cat "$SECRET_FILE") \
    OPENAI_API_BASE_URL=http://127.0.0.1:8200/v1 OPENAI_API_KEY=${AGENT_KEY:-agent-local} \
    ENABLE_OLLAMA_API=False ENABLE_TITLE_GENERATION=False ENABLE_FOLLOW_UP_GENERATION=False \
    ENABLE_TAGS_GENERATION=False ENABLE_AUTOCOMPLETE_GENERATION=False ENABLE_RETRIEVAL_QUERY_GENERATION=False \
    ENABLE_WEB_SEARCH=False DEFAULT_USER_ROLE=pending PATH=$W/bin:$PATH \
    setsid nohup "$W/bin/python" "$HERE/run_webui.py" > "$RUN/webui.log" 2>&1 &
  echo $! > "$RUN/webui.pid"; wait_http 127.0.0.1:8300/health webui 60;;
stop)
  stop_pg webui; stop_pg agent; "$HERE/serve_llm.sh" stop;;
status)
  for u in "llm 127.0.0.1:8100/health" "agent 127.0.0.1:8200/health" "webui 127.0.0.1:8300/health"; do
    set -- $u; curl -sf "$2" >/dev/null && echo "$1 up" || echo "$1 down"; done;;
esac
