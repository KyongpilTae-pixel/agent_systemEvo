#!/usr/bin/env bash
# 로컬 LLM(vLLM) 테스트 서버 기동/종료. 학습 큐와 GPU 공유 중 → 테스트할 때만 띄우고 끝나면 stop.
#   serve_llm.sh start [GPU] [MODEL]   (기본 GPU=가장 여유 큰 장, MODEL=qwen3-14b)
#   serve_llm.sh stop | status
# 주의(2026-09-30 P0 실측):
#   - PyPI 기본 vllm 은 CUDA13 → env llmserve 에 +cu129 휠 설치됨
#   - PATH 에 env bin 필수(ninja), FlashInfer 샘플러 JIT 실패 → VLLM_USE_FLASHINFER_SAMPLER=0
#   - util 은 총 메모리 대비. 0.50 은 KV 부족으로 기동 실패, 0.58(14.2GB) 에서 KV 12.5k 토큰
set -euo pipefail
E=$HOME/miniconda3/envs/llmserve
RUN=$HOME/llm_models/_run; mkdir -p "$RUN"
PORT=8100
cmd=${1:-status}

case "$cmd" in
start)
  MODEL=${3:-qwen3-14b}
  case "$MODEL" in
    qwen3-14b) NG=1; PATH_M=$HOME/llm_models/Qwen3-14B-AWQ; EXTRA="--max-model-len 12288 --gpu-memory-utilization 0.58 --tool-call-parser hermes --reasoning-parser qwen3";;
    # Qwen3.8 = 멀티모달(Qwen3_5ForConditionalGeneration) → 이미지/비디오 0 으로 비전 프로파일 생략. 도구호출 XML → qwen3_coder
    qwen3.8-27b) NG=2; PATH_M=$HOME/llm_models/Qwen3.8-27B-AWQ-INT4; EXTRA="--tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.58 --tool-call-parser qwen3_coder --max-num-seqs 4 --reasoning-parser qwen3 --limit-mm-per-prompt '{\"image\":0,\"video\":0}'";;
    gpt-oss-20b) NG=2; PATH_M=$HOME/llm_models/gpt-oss-20b; EXTRA="--tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.58 --max-num-seqs 4 --tool-call-parser openai --reasoning-parser openai_gptoss";;
    *) echo "unknown model $MODEL"; exit 2;;
  esac
  curl -s 127.0.0.1:$PORT >/dev/null 2>&1 && { echo "port $PORT 이미 사용 중 — 먼저 stop"; exit 4; }
  # 여유 메모리 큰 순으로 NG 장 선택 (인자로 "0" 또는 "0,1" 지정 가능)
  GPU=${2:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n "$NG" | cut -d, -f1 | paste -sd,)}
  [ "$GPU" = "auto" ] && GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n "$NG" | cut -d, -f1 | paste -sd,)
  for g in ${GPU//,/ }; do
    FREE=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$FREE" -lt 14500 ] && { echo "GPU$g free ${FREE}MiB < 14500 — 학습 잡 보호 위해 중단"; exit 3; }
  done
  eval "EXTRA_ARR=($EXTRA)"
  VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$E/bin/vllm" serve "$PATH_M" \
    --served-model-name "$MODEL" "${EXTRA_ARR[@]}" --enable-auto-tool-choice \
    --host 127.0.0.1 --port $PORT --api-key "${LLM_KEY:-local}" > "$RUN/vllm_$MODEL.log" 2>&1 &
  echo $! > "$RUN/vllm.pid"
  for _ in $(seq 1 180); do
    curl -sf -H "Authorization: Bearer ${LLM_KEY:-local}" 127.0.0.1:$PORT/v1/models >/dev/null && { echo "READY $MODEL on GPU$GPU"; exit 0; }
    kill -0 "$(cat "$RUN/vllm.pid")" 2>/dev/null || { echo "DIED — see $RUN/vllm_$MODEL.log"; exit 1; }
    sleep 5
  done; echo "timeout"; exit 1;;
stop)
  # setsid 로 띄웠으므로 프로세스 그룹 전체(API 서버 + EngineCore)를 종료
  P=$(cat "$RUN/vllm.pid" 2>/dev/null || true)
  if [ -n "$P" ] && kill -TERM -- "-$P" 2>/dev/null; then
    for _ in $(seq 1 30); do pgrep -g "$P" >/dev/null || break; sleep 1; done
    pgrep -g "$P" >/dev/null && kill -KILL -- "-$P"
    echo stopped
  else echo "not running (pid 파일 기준) — 잔존 확인: pgrep -af 'vllm serve'"; fi;;
status)
  curl -sf -H "Authorization: Bearer ${LLM_KEY:-local}" 127.0.0.1:$PORT/v1/models || echo "down";;
esac
