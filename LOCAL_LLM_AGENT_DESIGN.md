# 사내 분석 자동화 Agent 설계안 (로컬 LLM 기반)

> 작성 2026-09-30 · 상태: **설계 확정(테스트 단계)** · 결정 사항은 §9

## 0. 목표와 범위

| 항목 | 내용 |
|---|---|
| 용도 | **분석 자동화**: 팀원이 자연어로 요청하면 정본 코드(drastModules·drastUtils·claudeCode)를 도구로 호출해 평가·조회·리포트까지 수행 |
| 사용자 | 팀 공용 **웹 UI (Open WebUI)**. 이 서버는 **테스트 전용**, 운영 서버는 별도 |
| 대상 | **dRAST 2.5 + 3.0 모두** (도구에 `version` 인자 필수) |
| LLM | **오픈 가중치 모델을 받아 로컬 GPU에서 서빙**. 데이터는 서버 밖으로 나가지 않음 |
| 비범위(1차) | 모델 학습 자동 실행, 임의 코드 실행, 외부 API 호출 |

핵심 원칙: **LLM은 계산하지 않는다.** 숫자(EA·CA·PASS·VME)는 모두 정본 함수가 만들고, LLM은 "어떤 도구를 어떤 인자로 부를지"와 "결과 설명"만 담당한다. (자체 구현 금지·간이근사 금지 규약과 같은 이유)

## 1. 현재 서버 환경 (2026-09-30 실측)

| 자원 | 값 | 설계 영향 |
|---|---|---|
| GPU | RTX A5000 24GB × 4 (Ampere, CUDA 12.8) | FP8 네이티브 미지원 → **AWQ/GPTQ 4bit** 가 현실적 |
| GPU 사용 | 4장 모두 학습 큐가 ~9.5GB씩 사용 중 | **LLM 전용 GPU 확보 필요** (§9-1) |
| RAM / CPU | 503GB / 112코어 | 임베딩·RAG·평가는 CPU로 충분 |
| 네트워크 | huggingface.co·pypi.org 접근 가능 | 모델 다운로드·pip 설치 가능 |
| Docker | 설치됨, **권한 없음** | conda env + systemd --user / nohup 로 운영 |
| 기존 자산 | `agent_system/agents/*` (8 agent, tools() 메타), `sdk_tools.json`, `~/project/ai_agent_runner` (Anthropic/OpenAI API 호출형) | 도구 스키마 재사용. ai_agent_runner 의 provider 부분만 로컬로 교체 가능 |

## 2. 전체 구조

```
 [브라우저: 팀원]
      │ HTTPS(사내망) + 로그인
      ▼
 ┌──────────────┐    OpenAI 호환 API    ┌───────────────────────────┐
 │  Web UI       │ ───────────────────▶ │  Agent Server (FastAPI)    │
 │ (Open WebUI)  │ ◀─────────────────── │  - tool-calling 루프        │
 └──────────────┘   스트리밍 응답        │  - 권한/감사로그            │
                                        │  - 작업(Job) 관리           │
                                        └──┬─────────┬──────────┬───┘
                                           │         │          │
                          OpenAI 호환 API   │         │          │ subprocess (CPU)
                                           ▼         ▼          ▼
                              ┌────────────────┐ ┌────────┐ ┌─────────────────────┐
                              │ vLLM (GPU 1장)  │ │ RAG    │ │ 도구 = 정본 코드 래퍼  │
                              │ 오픈 LLM 서빙    │ │ 인덱스  │ │ drastModules/Utils   │
                              └────────────────┘ └────────┘ │ claudeCode 리포트     │
                                                            │ 큐 조회(읽기전용)      │
                                                            └─────────────────────┘
```

- **LLM 서빙과 Agent 를 분리**: 모델을 바꿔도 Agent 코드는 그대로 (OpenAI 호환 `/v1/chat/completions` 만 의존).
- Agent Server 자체가 OpenAI 호환 엔드포인트(`model="drast-agent"`)를 노출 → Open WebUI 가 이를 일반 모델처럼 붙여 씀. UI 교체도 자유.

## 3. LLM 선정

### 3.1 조건
1. **tool calling 품질** (가장 중요) 2. 한국어 3. 24GB 1장(4bit)에 적재 + 32k 이상 컨텍스트 4. **상업 사용 가능 라이선스**(Apache-2.0/MIT 우선)

### 3.2 후보 — **Apache-2.0 / MIT 만** (결정 §9-2, HF 실측 2026-09-30)

GPU 공유 조건(§9-1): 학습 잡이 장당 ~9.5GB 사용 → LLM 가용 **장당 ~13GB**.

| 후보 | 라이선스 | 가중치(4bit) | 배치 | 비고 |
|---|---|---|---|---|
| **Qwen3.8-27B** (dense, 2026-08) ★품질 후보 | Apache-2.0 | 21GB (`cyankiwi/Qwen3.8-27B-AWQ-INT4`, 커뮤니티 양자화) | GPU 2장 TP=2 (장당 ~10.5GB) | 최신 세대. 공식 AWQ 없음 → 운영 전 공식 FP8/자체 양자화 검토 |
| **Qwen3-14B AWQ** (2025-05) ★기준선 | Apache-2.0 | 10GB (공식) | GPU 1장 | 공유 조건에서 가장 안전 |
| gpt-oss-20b | Apache-2.0 | ~13GB (MXFP4) | TP=2 | Ampere 커널 지원 스모크 필요 |
| Qwen3.5-9B / 35B-A3B | Apache-2.0 | 12GB(AWQ) / ~20GB | 1장 / 2장 | 예비 |
| ~~Qwen3.8-Flash-Next~~ | other | 180B | | 라이선스·크기로 **제외** |
| ~~Gemma, EXAONE, Llama~~ | 별도 약관 | | | **제외** |

→ 테스트: Qwen3-14B(1장)로 파이프라인 먼저 → Qwen3.8-27B·gpt-oss-20b(각 2장) bake-off(§7) → 운영 모델 결정.

### 3.3 서빙: vLLM
- OpenAI 호환 API, tool-call 파서 내장(Qwen=`hermes`), 동시 사용자 배치 처리(팀 공용에 필요).
- 대안 Ollama 는 설치가 쉽지만 동시성·tool calling 제어가 약함 → 개인용 스모크에만.

```bash
# 별도 conda env (기존 qnt_algorithm env 오염 방지)
# 주의: PyPI 기본 vllm 0.30 은 CUDA 13(torch 2.13 cu13) → 드라이버 570(CUDA 12.8)에서 안 돔
#       → GitHub release 의 +cu129 휠 사용 (CUDA 12.x minor-version 호환)
conda create -n llmserve python=3.12 -y
pip install https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu129
# 공유 GPU: util 은 '총 메모리 대비' 비율 → 0.50 = 12GB (기동 시 free 가 이보다 커야 함)
CUDA_VISIBLE_DEVICES=<GPU> vllm serve ~/llm_models/Qwen3-14B-AWQ \
  --max-model-len 16384 --gpu-memory-utilization 0.50 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8100 --api-key $LLM_KEY
```
(모델 ID 는 예시 — 실제 레포명은 도입 시 확인)

## 4. 도구(Tool) 설계

도구는 **정본 함수의 얇은 래퍼**. 각 도구는 JSON schema + 권한 등급을 가진다.

| 등급 | 의미 | 예 |
|---|---|---|
| R (읽기) | 파일·상태 조회, 부작용 없음 | 바로 실행 |
| C (CPU 계산) | 평가·리포트 생성, 결과 파일 생성 | 바로 실행, 출력은 **agent 전용 디렉터리**에만 |
| W (쓰기/자원) | GPU 큐 등록, 공용 파일 수정 | **사용자 승인 버튼** 후 실행, 1차 범위 밖 |

### 4.1 1차 도구 목록

| 도구 | 등급 | 래핑 대상 |
|---|---|---|
| `search_knowledge(query)` | R | RAG: GLOSSARY·daily_reports·total_summary·설계문서·README |
| `glossary(term)` | R | `claudeCode/GLOSSARY.md` |
| `list_models()` / `resolve_model(name)` | R | `claudeCode/model_naming.py` (same_mic=deployed 등 별칭 포함) |
| `get_cell_metrics(model, organism, drug)` | R | 기존 평가 산출물(CSV) 조회 |
| `queue_status()` | R | `scripts/gpu_queue.tsv` + nvidia-smi 읽기 (priority 높을수록 우선 규칙 반영) |
| `evaluate_predictions(pred_file, dataset)` | C | drastModules `evaluateMics`·`getFDA_passFail`(5기준 minE) → EA/CA/ME/VME/PASS 표 |
| `evaluate_reproducibility(pred_file)` | C | `drastUtils/reproducibility.py` |
| `compare_models(a, b, dataset)` | C | 셀 단위 승/패·PASS 전환 양방향·VME 변화 **강제 포함** |
| `build_cell_dossier(gram, drug, models)` | C | `claudeCode/cell_dossier.py` |
| `make_report(kind, inputs)` | C | report-builder 관행(공통 CSS·검증) HTML 생성 |

### 4.2 도메인 규약을 도구에 박아 넣기
LLM 이 규약을 "기억"하게 하지 않고 **도구가 강제**한다.
- 평가표 출력에 표본수·EA·CA·정제셋 항상 포함
- 비교 결과는 셀(organism×drug) 단위 표가 없으면 반환 거부
- 패널 키 (project_id, sample_id, antimicrobial) 고정 — sample_id+drug 합산 버그 차단
- 유의성 검정 단위=셀 (웰 pooling 금지)
- 3.0 / 2.5 인자 필수 (GF 규칙·breakpoint·control 규칙이 다름)

### 4.3 시스템 프롬프트 + 지식
- 시스템 프롬프트: 역할, 용어 요약(VME/ME/EA/CA/PASS), "숫자는 도구 결과만 인용, 없으면 모른다고 답함", 추측/검증 구분 표시.
- RAG: 임베딩 **bge-m3**(다국어, CPU 가능) + BM25 하이브리드, 저장소는 로컬 SQLite/FAISS. 원천 약 2MB 규모(일일보고 74건·GLOSSARY·요약) → 인덱싱 수 분. 야간 cron 재색인.

## 5. Agent Server

- FastAPI, 자체 구현 tool-calling 루프 (LangChain 등 대형 프레임워크 미사용 → 코드 200~300줄, 감사 용이).
- 루프: 메시지 → LLM → tool_calls 실행 → 결과 주입 → 반복 (최대 N=10회, 도구 타임아웃).
- 오래 걸리는 C 등급 도구는 **Job** 으로 비동기 실행 → UI 에 진행 상태·완료 시 링크.
- 결과물: `agent_system/web_agent/output/<user>/<job_id>/` (공용 산출물 디렉터리 오염 금지).
- 모든 요청·도구 호출·인자·결과 요약을 JSONL 감사 로그로 남김.

## 6. 보안·운영

| 항목 | 방안 |
|---|---|
| 데이터 반출 | LLM·임베딩·UI 모두 로컬, 외부 호출 없음 |
| 접근 | vLLM 은 127.0.0.1 바인딩, Agent/UI 만 사내망 노출. Open WebUI 계정(관리자 승인 가입) |
| 파일 접근 | 도구별 경로 allowlist, 쉘·임의 코드 실행 도구 없음(1차) |
| 자원 | **테스트 기간 공유**: vLLM 은 util 0.50 상한, 테스트할 때만 기동·끝나면 종료. 큐 러너는 vLLM 메모리를 모르므로 학습 잡 OOM 위험 → 기동 전 여유 확인, OOM 발생 시 LLM 쪽을 내림(학습 우선) |
| 기동 | systemd --user 또는 @reboot cron (기존 ensure_aux_watchers 방식과 동일), 헬스체크 실패 시 재기동 + 메일 |
| 모델 교체 | 모델 가중치는 `~/llm_models/` 에 버전별 보관, 서빙 설정만 바꿔 롤백 |

## 7. 평가 (모델 선정·회귀 테스트)

**골든셋 40~60문항**을 과거 실제 작업에서 만든다 (정답이 이미 확정된 것).
- 조회형: "운영 권고 recipe 와 그 PASS/VME 는?" → routed_iso_t65, 103, 1.99%
- 도구선택형: "cdn_6h 와 cdn_full 을 2.5 임상에서 비교해줘" → `compare_models` 를 올바른 인자로 호출했는가
- 규약형: 웰 pooling p값을 요구하는 질문에 셀 단위로 답하는가
- 거절형: 결과 파일이 없을 때 숫자를 지어내지 않는가

지표: 도구 선택 정확도 · 인자 정확도 · 최종 답 정확도 · **숫자 환각률(0 목표)** · 응답 시간. 모델 3종 bake-off 후 확정, 이후 모델 교체 때마다 회귀 테스트.

## 8. 단계별 계획

| 단계 | 내용 | 산출물 | 완료 기준 |
|---|---|---|---|
| P0 | llmserve env, 후보 3종 다운로드·vLLM 스모크, 골든셋 초안 | 서빙 스크립트, 골든셋 | 3종 모두 tool call 왕복 성공 |
| P1 | Agent Server + R 등급 도구 + RAG, 모델 bake-off | 1차 모델 확정 리포트 | 골든셋 조회형 ≥ 90%, 숫자 환각 0 |
| P2 | C 등급 도구(평가·비교·dossier·리포트), Job, Open WebUI 연결 | 팀 공개 베타 | 정본 결과와 sample-by-sample 일치 |
| P3 | W 등급(승인 후 GPU 큐 등록), 필요 시 샌드박스 코드실행 | | 승인 흐름·감사로그 검증 |

## 8.1 P0 결과 (2026-09-30)

| 항목 | 결과 |
|---|---|
| 환경 | `llmserve`(py3.12, vllm 0.30.0+cu129, torch 2.13+cu129) · `openwebui`(py3.11, open-webui 0.11.4) 설치 완료 |
| 가중치 | `~/llm_models/`: Qwen3-14B-AWQ 9.4G · Qwen3.8-27B-AWQ-INT4 20G · gpt-oss-20b · bge-m3 |
| 기동 | `agent_system/web_agent/serve_llm.sh start|stop|status` — 여유 최대 GPU 자동 선택, 여유 <14.5GB 면 기동 거부 |
| 메모리 | Qwen3-14B: util 0.58 = 12.9GB 점유, KV 12.5k 토큰(동시 1건). 0.50 은 KV 부족으로 기동 실패 |
| 스모크 | `smoke_llm.py` 4/4 통과 (한국어·도구선택·인자·도구결과 인용). 응답 3.5~8초 |

해결한 설치 함정: PyPI vllm=CUDA13(드라이버 570 불가)→cu129 휠 · env PATH 미포함 시 ninja 없음 · FlashInfer 샘플러 JIT 실패→`VLLM_USE_FLASHINFER_SAMPLER=0` · nohup 부모만 kill 하면 EngineCore 가 GPU 를 쥔 채 남음→setsid+프로세스그룹 종료 · Qwen3 `<think>` 노출→`--reasoning-parser qwen3`.

설계에 반영할 관찰:
1. **도메인 문맥 없이는 오답**: "VME 가 뭐야?" → 컴퓨터 버스 규격(VMEbus)으로 답함. 시스템 프롬프트 용어 요약 + `glossary`/RAG 가 필수임을 확인.
2. **LLM 이 파생 수치를 직접 계산**: PASS 103/313 → "32.9%" 를 스스로 계산. 맞았지만 원칙(§0) 위반 → 도구가 비율까지 계산해 반환하고, 프롬프트에 "도구에 없는 수치 계산 금지" 명시.
3. 공유 GPU 동시성 1건 → 팀 베타 전에는 전용 GPU 또는 운영 서버 필요.

## 8.2 P1 결과 (2026-09-30) — Agent 서버 + 도구 + Open WebUI + 모델 bake-off

**구성** (`agent_system/web_agent/`)
- `serve_stack.sh start|stop|status` — vLLM(8100) → Agent(8200, OpenAI 호환 `drast-agent`) → Open WebUI(8300). 전부 127.0.0.1, 브라우저는 VS Code 포트포워딩.
- 도구 7종(R 등급): `glossary` · `search_knowledge`(BM25, 문서 966조각) · `queue_status` · `list_models` · `cell_metrics` · `reproducibility` · `compare_models`(셀 단위, 필드명에 모델명).
- 감사 로그 `web_agent/logs/audit_YYYYMMDD.jsonl`. 골든셋 `eval/golden.json`(16문항) · 채점 `python -m agent_system.web_agent.eval.run_golden`.

**정본 대조 검증**: 2.5 셀 PASS 를 night_axes25 경로 재사용으로 계산 → **111/111 모델 clin_pass·clin_cells 일치**. 과정에서 `trainall_*`(학습 패널 포함 추론)이 섞이는 경로를 발견해 제외.

**bake-off** (도구 수정 반영 후, 16문항)

| 모델 | 자동채점 | 평균 응답 | GPU | 사람 검토(비교 답변 방향) |
|---|---|---|---|---|
| **Qwen3.8-27B (AWQ, TP=2)** ★기본 | **16/16** | 16.0s | 2장 × ~13.5GB | 정확 (PASS 전환 양방향·VME 방향 모두 맞음) |
| gpt-oss-20b (MXFP4, TP=2) | 15/16 | **4.4s** | 2장 | EA 우위 셀 수 방향 뒤바꿈, VME 차이 자체 계산 |
| Qwen3-14B (AWQ) | 14/16 | 7.7s | 1장 × 12.9GB | PASS 전환 방향 뒤바꿈 + "baseline VME 더 적음" 오결론 → **탈락** |

- 되묻기(세대 불명): gpt-oss 만 정확히 되물음. Qwen3.8 은 3.0 으로 답하고 2.5 도 제안.
- **자동채점 한계**: 비교 답변의 방향 오류(A/B 뒤바뀜)를 못 잡음 → 다음 단계에서 방향 검사(수치-모델명 쌍 대조) 추가 필요.

**설치·기동 함정 추가**: Open WebUI 0.11.4 는 langchain/scipy 등 로드 후 pyarrow 를 불러오면 세그폴트 → `run_webui.py` 에서 pyarrow 선로드 · Qwen3.8 은 멀티모달+하이브리드 → `--limit-mm-per-prompt image/video 0`, `--max-num-seqs 4`(상태캐시 6GB 절감), 도구 파서 `qwen3_coder` · gpt-oss 는 `openai`/`openai_gptoss` 파서.

**공유 GPU 부담**: 27B/gpt-oss 모두 2장 사용 시 해당 GPU 여유 < 1GB → 학습 잡 메모리 증가 시 OOM 위험. 테스트할 때만 기동.

## 9. 결정 사항 (2026-09-30)

1. **GPU**: 당분간 학습 큐와 **공유**. 이 서버는 테스트 전용, 운영은 별도 서버.
2. **라이선스**: **Apache-2.0 / MIT 모델만**.
3. **UI**: **Open WebUI**.
4. **범위**: **2.5 + 3.0 모두** 1차 포함 (GF 규칙·control 규칙·breakpoint·패널 키 분기를 도구가 처리).

## 10. 운영 배포 전제 (2026-10-06 확정)

| 항목 | 내용 | 설계 영향 |
|---|---|---|
| 실행 위치 | 이 서버 = 실험, 운영 = **별도 PC (Windows)** | vLLM 은 Windows 네이티브 미지원 → 운영 서빙은 **Ollama(GGUF)**. Agent 는 OpenAI 호환만 의존하므로 `LLM_BASE` 만 교체 |
| GPU | **RTX A5000 24GB 1장** | TP=2 구성(§8.2 기본) 사용 불가. 후보 = Qwen3.8-27B Q4_K_M 16.5GB · gpt-oss-20b Q4_K_M 11.6GB (둘 다 Apache-2.0, `unsloth/*-GGUF`) |
| 사용자 | **다른 팀까지** | 문서 Q&A 가 중심. 도구 묶음 분리(`TOOL_MODULES`: 공통=knowledge / 알고리즘 팀=ops,metrics). 문서별 접근 권한 필요 |
| 데이터 | 운영 PC 에서 **이 서버 파일을 볼 수 있음** | 도구는 파일 읽기만. 실행이 필요한 것은 이 서버가 `web_agent/exports/` 로 내보냄 |

**내보내기**: `export_snapshots.py`(qnt_algorithm env, 약 2분) → `exports/cells25.parquet`(111모델·19,818행) · `queue_status.txt` · `manifest.json`. 도구는 스냅샷 우선, 없으면 직접 계산. 검증: 스냅샷 기준 night_axes25 대조 **0/111 불일치**, 직접 계산과 동일.

**경로 환경변수**(운영 PC 에서 매핑 드라이브로 지정): `QNT_ROOT` `R25_NEWMODEL` `F30_DEPLOY_REPRO` `EXTRA_KNOWLEDGE` `LLM_BASE` `LLM_MODEL` `TOOL_MODULES`.

**남은 일**: ① GGUF 2종을 Ollama 1장 구성으로 골든셋 재채점(양자화 포맷이 달라 §8.2 순위는 참고용) ② bash 기동 스크립트의 Windows 대체(ps1) ③ 스냅샷 주기 실행(cron) ④ 타 팀 문서 수집·권한 구조 ⑤ 의미 기반 검색(bge-m3).
