# Human MIC Labeling (우리쪽 관리본)

sample_id / antimicrobial 별 **휴먼 MIC**를 라벨링하는 웹 UI.

> **원본** = `gyuyoung` 님의 `/home/gyuyoung/project/lab/human_labeling` (2026-09-08 우리쪽 복사).
> 코드는 그대로 두고 **경로·포트만** 우리 환경으로 바꿨다. 원본과 갈라지므로 상류 변경은 수동 반영한다.

## 실행
```bash
cd /home/kptae/project/qnt_algorithm/agent_system/human_labeling
/home/kptae/miniconda3/envs/qnt_algorithm/bin/python app.py   # conda env: qnt_algorithm
# 브라우저에서 http://localhost:5058 접속
```
★포트는 **5058** — 원본(5057)과 나란히 띄울 수 있게 비켜 놓았다.
의존성: `pandas`, `numpy`, `Pillow`, `safetensors` (표준 라이브러리 HTTP 서버라 Flask 불필요)

## 사용법
- **왼쪽 메뉴바**: 4단 폴더 트리 + 검색 + 진행률(완료/전체).
  ```
  FDA_Clinical / Reproducibility / Sample_Stability  →  antimicrobial  →  organism_group  →  microbial_id  →  샘플
  ```
  폴더를 클릭해 펼치고 접으며, 각 폴더 오른쪽에 `완료/전체` 개수가 표시된다. 완료 항목은 초록 점과 MIC 값 표시.
  목록을 스크롤해도 폴더 헤더는 위에 고정돼(sticky) 지금 보고 있는 샘플이 어느 약제·균종인지 항상 보인다.
- **그룹핑 순서**: 진행률 아래 `약제 → 균종` / `균종 → 약제` 버튼(단축키 `g`)으로 트리 순서를 바로 바꾼다.
  ```
  약제 → 균종 :  … → antimicrobial → organism_group → microbial_id → 샘플   (기본)
  균종 → 약제 :  … → organism_group → microbial_id → antimicrobial → 샘플
  ```
  최상위(FDA_Clinical / Reproducibility / Sample_Stability)는 고정이고, `microbial_id`는 `organism_group`의 하위 분류라 항상 그 바로 뒤에 붙는다.
  바꿔도 보던 샘플은 그대로 열려 있고, 선택은 브라우저에 기억된다. `d`/`a` 이동 순서와 프리페치도 바뀐 순서를 따른다.
- **가운데**: 상단에 `sample_id · microbial_id · antimicrobial`. 그리드는 왼쪽에 농도(control은 `Cont`), 위에 `Time 0~6`.
- **오른쪽 라디오**: 각 농도 행마다 라디오 1개 + 맨 아래 `>=32`(전 농도 성장) 옵션. 선택 즉시 자동 저장.
- **셀 확대**: 그리드의 이미지를 클릭하면 크게 뜬다(원본 224x224). 확대 중에는 방향키로 이웃 셀(←→ 시간, ↑↓ 농도)을 볼 수 있고,
  `Esc`나 화면 클릭으로 닫는다. 확대 중에는 라벨 단축키가 먹지 않아 실수로 저장될 일이 없다.
- **BMD MIC 토글**: 상단 `BMD MIC` 버튼(단축키 `b`). ★**기본 표시**(2026-09-08 변경).
  원본은 감춤이 기본이었다 — 라벨링이 BMD 에 끌려가지 않게 하려던 것이다.
  지금 용도가 “bmd 가 틀렸나, 이미지가 이상한가”를 가리는 **대조 작업**이라 보이는 쪽으로 바꿨다.
  ⚠**독립적인 사람 판독이 필요하면 `b` 로 끄고 라벨한다** — 켠 채로 매기면 BMD 에 앵커링된다.
  켜면 상단에 BMD 값이 뜨고, `bmd_mic_order`에 해당하는 농도 행이 보라색 `BMD` 태그와 테두리로 표시된다. 켠 상태는 브라우저에 기억된다.
- **단축키**: `f` **서브셋 필터 토글**, `d`/`a` 다음·이전 샘플, `1~9` 농도 라디오 선택, `0` 전 농도 성장, `b` BMD MIC 표시/감춤, `g` 그룹핑 순서 전환.

## 특정 샘플 바로 열기 (주소 공유)
주소창은 항상 현재 보고 있는 샘플을 반영한다. 그 주소를 복사해 두거나 남에게 주면 같은 화면이 바로 열린다.

```
http://localhost:5057/?project_id=202308_FDA_Clinical_USA&sample_id=1010-QMX-IUP&antimicrobial=PTZ
```
- `project_id`는 생략해도 된다 (`sample_id` + `antimicrobial` 조합이 유일하면 그대로 열린다).
- 대소문자는 구분하지 않는다.
- 찾지 못하면 토스트로 알리고 평소처럼 첫 미완료 샘플을 연다.

## MIC 기록 규칙
- 최저농도(0.06) 선택 → `<=0.06`
- 그 외 농도 선택 → 해당 농도 값 (예: `0.5`)
- `>=32` 선택 → `>=32` (최고농도 16.0 초과 = 전 농도 성장)

## 입력 데이터 — ★3개 데이터셋

`app.py` 상단 `SOURCES`. **사이드바 최상위 폴더로 갈라져** 한 서버에서 셋 다 라벨링한다.
셋은 [[dataset_glossary]] 의 3개 평가축과 같다 — 임상 · 재현성 · 재시험 안정성.

| 최상위 폴더 | 건수 | df (parquet) · allinfo (csv) — 모두 `/home/kptae/data/allinfo/` 아래 |
|---|---|---|
| **FDA_Clinical** | 17,993 | `new_fda2023/fda_clinical_MEV_truncation_20260522_exceptGF,JinhaTE_df.parquet` · `new_fda2023/202510_FDA_Clinical_USA_allInfo_MEV_truncation_260401.csv` |
| **Reproducibility** | 19,008 | `analytical/FDA_Analytical_reproducibility_exceptGN26_df_260624.parquet` · `analytical/FDA_Analytical_reproducibility_MEVtruncation_allInfo_260416.csv` |
| **Sample_Stability** | 1,831 | `analytical/FDA_Analytical_sample_stability_df_260723_DelOutlier_with_BMD.parquet` · `analytical/FDA_Analytical_sample_stability_allInfo_new_260723.csv` |

| **dRAST2.5_d170** | 599 | `d25/d170_label_source.parquet` (`build_source25.py` 로 생성) |

**총 39,431건** — 단 ★**기본은 2.5(599건)만 보인다.**

3.0 세 소스는 `SOURCES` 의 `enabled: False` 로 **감춰 뒀다**(2026-09-08 지시).
지금 검토 대상이 2.5 d170 뿐인데 3.0 38,832건이 섞이면 트리가 묻히기 때문이다. 데이터는 지우지 않았다.
되살리려면 `SHOW_30=1 python app.py`.

### ★2.5(d170)는 형식이 다르다
| | 3.0 세 소스 | **2.5 d170** |
|---|---|---|
| 이미지 | 패널당 `.safetensors` 한 덩어리 `(N,1,224,224)` | **웰마다 PNG 7장** |
| 셀 만들기 | 텐서 블록을 잘라 PNG 저장 | **PNG 를 그대로 복사**(224×119, 리사이즈 안 함) |
| 균종·BMD | allinfo CSV 를 merge | d170 CSV 에 이미 있음(merge 생략) |
| `bmd_mic_order` | df 컬럼 | **`bmd_mic`+농도배열로 계산**(0=최저 · 14=전 농도 성장) |

`SOURCES` 의 `loader` 가 `"png25"` 면 이 경로를 탄다. `allinfo: None` 이면 merge 를 건너뛴다.
소스 갱신은 `python build_source25.py`.

## ★TE(기술오류) 모델 결과 표시 — 2.5 전용

격자 **행 라벨 옆에 배지**로 뜬다. `기포 0.97` / `필름 0.62` 처럼 값까지 보이고,
마우스를 올리면 작동점이 같이 뜬다. 강한 신호가 있는 행은 농도 라벨에 점선 밑줄이 붙는다.

| 표시 | 조건 | 뜻 |
|---|---|---|
| 빨강 배지 | `bubble ≥ 0.41` · `film ≥ 0.31` | 검증셋에서 잡은 **작동점 초과** |
| 주황 배지 | `≥ 0.15` | 약한 신호 — 참고만 |
| 없음 | `< 0.15` | 조용함 |

- 모델 = **2.5 전용 TE**(`newmodel/trained_te25/..._h119_ep30_d25`, 0.6M · 2.3MB).
  3.0 TE(147M)를 2.5 에 그냥 쓰면 종횡비·도메인이 어긋나서, 2.5 로 따로 학습한 것을 쓴다.
- ★**control 행도 채점한다** — control 에 기포가 끼면 정규화가 흔들려 패널 전체가 오염된다.
- 점수 생성 = `drast_25_lrcn` 에서 `python -m newmodel.infer_te25_panels` →
  `newmodel/data/te25_panel_scores.csv` → `build_source25.py` 가 parquet 에 실어 준다.
  점수 파일이 없으면 배지 없이 그대로 뜬다.
- ⚠**모델 출력이지 판정이 아니다.** 사람 라벨 22웰 검증에서 양성 5/5 검출 · 오검출 1 이었지만
  표본이 작다. 배지는 "여기 한 번 보라"는 신호이고, TE 체크박스는 사람이 직접 켠다.

## ★서브셋 — "이것만 보기"

사이드바 위 **`보기`** 드롭다운으로 검토 목록만 걸러 본다. 단축키 **`f`** 는 전체 ↔ 직전 서브셋 토글.
필터는 브라우저에 기억되고, **`d`/`a` 이동·프리페치·진행률이 모두 걸러진 범위를 따른다**
(진행률을 전체 39,431 기준으로 두면 599건 목록을 도는 동안 바늘이 안 움직인다).

| 서브셋 | 건수 | 무엇인가 |
|---|---|---|
| `op_only` **운영만 맞힘(EN)** | 218 | 운영은 EA 통과인데 구조모델이 틀린 패널. 구조 교체의 **실제 비용** |
| `allwrong` **전 구조 실패 웰 포함** | 425 | 18개 구조가 **전부** 틀린 웰(628개)이 든 패널. bmd↔이미지 방향 충돌 층 |
| `te_suspect` **TE 기술오류 의심** | 16 | 2.5 전용 TE 모델이 버블/필름으로 지목한 웰이 든 패널 |

**새 목록을 추가하려면** `subsets/<id>.csv` 에 `project_id,sample_id,antimicrobial` 세 컬럼으로 떨구고
서버를 재기동하면 자동으로 잡힌다. 이름은 `app.py` 의 `SUBSET_LABELS` 에 넣는다(없으면 id 가 그대로 뜬다).
⚠서브셋은 **패널 단위**다 — 웰 단위 목록(628 웰)은 그 웰이 속한 패널로 접혀 425개가 된다.

⚠**safetensors 이미지는 복사하지 않았다.** parquet 의 `image_safetensors_path` 가
`/home/gyuyoung/safetensors/...` 를 가리키고 **읽기 권한이 있어 그대로 참조**한다(전체 ~74GB).
원본이 옮겨지거나 지워지면 이미지가 안 뜬다 — 그때 경로를 우리 볼륨으로 바꾼다.

`organism_group` / `microbial_id`는 df에 없으므로 allinfo에서 `(project_id, sample_id)` 기준으로 merge 해 온다.
allinfo가 수백 MB라 필요한 4개 컬럼만 뽑아 `.cache/<id>_bugmap.parquet`에 캐싱하며, allinfo 파일이 더 새로우면 자동으로 다시 만든다.
한 sample에 값이 여러 개면 임의로 고르지 않고 ` / `로 이어붙여(예: `Klebsiella pneumoniae / Klebsiella variicola`) 모호함이 트리에 그대로 보이게 한다.

## 출력
**균종·약제 구분 없이 `image_mic_labels.csv` 파일 하나**로 관리된다. 라벨할 때마다 갱신되며 재실행 시 이어서 작업 가능.

| 컬럼 | 설명 |
|---|---|
| project_id | 프로젝트 ID (`202308_FDA_Clinical_USA` / `reproducibility`) |
| sample_id | 샘플 ID |
| organism_group | 균 그룹 (allinfo merge) |
| microbial_id | 균종 (allinfo merge) |
| antimicrobial | 항생제 |
| image_mic | Image MIC (예: `<=0.06`, `0.5`, `>=32`) |
| image_mic_order | MIC 농도 순서 (0=최저농도, `all_growth`=14) |
| ambiguous | ambiguous 플래그 (1/0) |
| TE | Technical Error 플래그 (1/0) |

## 폴더 구성
- `app.py`, `index.html` — 서버 / UI
- `prefetch.py` — 대량 프리페치
- `build_source25.py` — 2.5 소스·서브셋 생성
- `subsets/*.csv` — 서브셋 정의
- `image_mic_labels.csv` — 라벨 출력 (단일 파일). **빈 상태에서 시작한다** —
  원본에 있던 gyuyoung 님 라벨은 가져오지 않았다(라벨러가 섞이면 안 된다).
  필요하면 `/home/gyuyoung/project/lab/human_labeling/image_mic_labels.csv` 를 참조용으로 따로 읽는다.

## 첫 로드가 느릴 때 (프리페치)
샘플을 처음 열면 safetensors에서 셀 PNG를 뽑아내야 한다. 추출 자체는 샘플당 약 0.3초지만
**머신이 바쁘면(load average가 높으면) 10초 이상 걸린다.** 한 번 캐시되면 이후에는 즉시 뜬다.

- **자동**: 샘플을 열면 서버가 뒤이어 나올 `PREFETCH_AHEAD`개(기본 8)를 백그라운드에서 미리 추출한다.
  `d`로 순서대로 넘기며 라벨링하면 기다릴 일이 거의 없다.
  환경변수 `PREFETCH_AHEAD`, `PREFETCH_WORKERS`(기본 2)로 조절한다.
- **수동(대량)**: 작업할 범위를 미리 통째로 채워둘 때는 `prefetch.py`를 쓴다. 서버를 켜 둔 채로 돌려도 된다.

```bash
python prefetch.py --source fda_clinical --drug MP --workers 4   # MP 약제 전체
python prefetch.py --microbial "K. pneumoniae" --todo-only       # 미완료분만
python prefetch.py --drug MP PTZ --dry-run                       # 개수·용량만 확인
```
`--source / --drug / --organism / --microbial / --todo-only / --limit / --workers / --dry-run` 조합으로 범위를 정한다.
4프로세스 기준 초당 6~7개 정도 나오며, 전체 37,001건을 다 받으면 약 74GB · 1.5시간 규모다.

## 캐시 (`/home/kptae/human_labeling_cache/`)
용량이 커서 프로젝트 폴더가 아닌 큰 볼륨(13T)에 둔다. 지워도 자동으로 다시 만들어진다.

| 경로 | 내용 |
|---|---|
| `cells/<sample>__<약제>/` | 셀 이미지 PNG. 샘플을 처음 열 때 생성되며 **샘플당 약 2MB** |
| `bugmap/<id>_bugmap.parquet` | allinfo에서 뽑은 균종 매핑 (sample 단위) |
| `bugmap/<id>_bmdmic.parquet` | allinfo에서 뽑은 BMD MIC (sample x 약제 단위) |

두 매핑은 allinfo를 한 번만 읽어 함께 만들며, allinfo 파일이 더 새로우면 자동 재생성된다.

### BMD MIC 표기 주의
화면에 뜨는 값은 **allinfo `bmd_mic` 원본 표기**이고, 행 하이라이트는 df의 `bmd_mic_order`를 따른다.

- **CAZC / CTXC는 BMD를 아예 표시하지 않는다** (`app.py`의 `BMD_HIDDEN_DRUGS`).
  이 두 약제는 `bmd_mic_order`가 전부 임의값 14로 덮여 있고 `bmd_mic`도 `POS`/`NEG`/`ND`라 참고할 수 없다.
- 나머지 약제 중 조합약제는 `4/2`(약제/억제제) 형태로 적혀 있고, 분자가 실제 MIC이며 order도 정상이다.
- 측정 범위를 벗어나면 `>8`(패널 표기로는 `>=16`)처럼 적혀 있어 값과 하이라이트 위치가 어긋나 보일 수 있다.

## 설정 (app.py 상단)
`SOURCES`(입력 데이터), `CSV_PATH`(출력), `CACHE_ROOT`(캐시 위치, 환경변수 `CACHE_ROOT`로 변경 가능),
`PORT`(기본 **5058**, 환경변수 `PORT`로도 지정 가능).
