#!/usr/bin/env python3
"""Human MIC labeling server.

safetensors 이미지(농도 x Time 그리드)를 브라우저로 보여주고,
sample_id / antimicrobial 별 휴먼 MIC를 라디오로 입력받아 CSV로 저장한다.

의존성: 표준 라이브러리 + pandas + numpy + PIL + safetensors (Flask 불필요)
실행:  python3 app.py  ->  http://localhost:5057
"""

import csv
import gzip
import json
import os
import queue
import re
import shutil
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
from PIL import Image
from safetensors import safe_open

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# 캐시(약 64GB)는 /208_data 가 아니라 홈 볼륨에 둔다. 환경변수 CACHE_ROOT 로 변경 가능.
#
# /208_data(sdb) 는 여러 사람이 함께 쓰는 회전식 어레이라 남의 작업이 몰리면 %util 이 99% 까지
# 올라간다. 샘플 하나가 셀 PNG 49~70개, 즉 랜덤 읽기 49~70회라 이때 직격탄을 맞는다.
# 실측(무작위 12샘플 콜드 읽기):
#     /208_data (sdb) : 중앙값 132ms, 최악 1252ms
#     홈 볼륨  (sda) : 중앙값  12ms, 최악   15ms
# 평균보다 꼬리가 중요하다 — "가끔 답답한" 증상의 정체가 저 1252ms 였다.
CACHE_ROOT = os.environ.get("CACHE_ROOT", "/home/kptae/human_labeling_cache")
CELLS_DIR = os.path.join(CACHE_ROOT, "cells")
CACHE_DIR = os.path.join(CACHE_ROOT, "bugmap")
INDEX_HTML = os.path.join(HERE, "index.html")
PORT = int(os.environ.get("PORT", "5058"))

# 셀 이미지 URL 에는 ?v=<.done mtime> 캐시버스터가 붙으므로(ensure_cells 참고) 오래 캐시해도
# 안전하다. 매일 쓰는 도구라 하루(86400)마다 전부 다시 받는 건 낭비다.
IMG_CACHE_CONTROL = "public, max-age=31536000, immutable"

# 라벨 대상 데이터. label 이 사이드바 최상위 폴더 이름이 된다.
# organism_group / microbial_id 는 allinfo 에서 merge 해 온다 (parquet 에 있어도 allinfo 를 따른다).
SOURCES = [
    {
        "id": "fda_clinical",
        "enabled": False,   # ★3.0 — 지금은 감춤(SHOW_30=1 로 되살린다)
        "label": "FDA_Clinical",
        "parquet": "/home/kptae/data/allinfo/new_fda2023/"
        "fda_clinical_MEV_truncation_20260522_exceptGF,JinhaTE_df.parquet",
        "allinfo": "/home/kptae/data/allinfo/new_fda2023/"
        "202510_FDA_Clinical_USA_allInfo_MEV_truncation_260401.csv",
    },
    {
        "id": "reproducibility",
        "enabled": False,   # ★3.0 — 지금은 감춤(SHOW_30=1 로 되살린다)
        "label": "Reproducibility",
        "parquet": "/home/kptae/data/allinfo/analytical/"
        "FDA_Analytical_reproducibility_exceptGN26_df_260624.parquet",
        "allinfo": "/home/kptae/data/allinfo/analytical/"
        "FDA_Analytical_reproducibility_MEVtruncation_allInfo_260416.csv",
    },
    {
        # ★2.5(d170) — 3.0 세 소스와 입력 형식이 다르다.
        #   3.0 = 패널당 safetensors 한 덩어리 / 2.5 = 웰마다 PNG 7장.
        #   그래서 loader 를 나누고, allinfo merge 도 하지 않는다(d170 CSV 에 균종·bmd 가 이미 있다).
        "id": "d25",
        "label": "dRAST2.5_d170",
        "loader": "png25",
        "parquet": "/home/kptae/data/allinfo/d25/d170_label_source.parquet",
        "allinfo": None,
    },
    {
        "id": "sample_stability",
        "enabled": False,   # ★3.0 — 지금은 감춤(SHOW_30=1 로 되살린다)
        "label": "Sample_Stability",
        "parquet": "/home/kptae/data/allinfo/analytical/"
        "FDA_Analytical_sample_stability_df_260723_DelOutlier_with_BMD.parquet",
        "allinfo": "/home/kptae/data/allinfo/analytical/"
        "FDA_Analytical_sample_stability_allInfo_new_260723.csv",
    },
]

# ★어떤 소스를 띄울지. 기본은 2.5 만 — 지금 검토 대상이 2.5 d170 뿐이라
#   3.0 38,832건이 섞이면 트리가 묻힌다(2.5 는 599건). 데이터는 지우지 않고 감추기만 한다.
#   되살리기: SHOW_30=1 python app.py
SHOW_30 = os.environ.get("SHOW_30", "0").strip().lower() in ("1", "true", "yes", "y")
SOURCES = [s for s in SOURCES if s.get("enabled", True) or SHOW_30]


# BMD를 화면에 표시하지 않을 약제.
# bmd_mic_order가 임의값(0/14)으로만 채워져 있고 bmd_mic도 POS/NEG/ND 같은
# 스크리닝 판정이라 라디오 눈금과 대응되지 않는다. 참고할 수 없는 값이므로
# 아예 내보내지 않는다.
BMD_HIDDEN_DRUGS = {"CAZC", "CTXC", "HLG", "HLS", "CXS"}

# ★서브셋 — "이것만 보기" 로 거르는 이름표. subsets/<id>.csv 에
#   project_id,sample_id,antimicrobial 세 컬럼. 패널 단위이고 여러 개가 겹칠 수 있다.
#   파일을 넣기만 하면 자동으로 잡히므로, 새 검토 목록이 생기면 CSV 만 떨구면 된다.
SUBSET_DIR = os.path.join(HERE, "subsets")
SUBSET_LABELS = {
    # ────────── VME (임상 안전성 — 내성균을 놓치는 오류) ──────────
    #   2026-09-14 전면 재산출(newmodel/build_vme_subsets25.py). 셋을 고쳤다.
    #     ① 옛 목록은 dataset_170 **정정(9/11) 이전** 예측 산출이었다.
    #     ② 운영축을 `lrcn_gng`(커버리지 95.6%·옛 ref 탓 5,451패널로 묶임) →
    #        **`drast_mic`**(순차 판정의 끝·커버리지 100%)으로 교체.
    #     ③ 학습 sample 제외 **철회** — 운영축은 우리 학습셋을 모른다. 분모가 같아야 한다.
    #   분모 5,357 → 15,216패널 · 표적 3건 → 44건.
    #   (운영 drast_mic VME 109 · 우리 cdn_6h 135 · 공유 91 · 우리만 44 · 운영만 18)
    "vme_unflagged": "VME ★★신호 미포착 — 최우선",
    # ★2026 신규 패널(Enterobacteriaceae 18약제)에 드는 것만 — 앞으로의 실제 표적.
    #   정본 = release_25/dRASTBreakpoints_2026 new panel.csv (사용자 제공 2026-09-14).
    #   CIP·CZ·IP·TS 가 빠져 21건 → 10건이 된다. CIP 7건이 통째로 빠지는 게 크다.
    "vme_new_panel2026": "VME ★★★개선 표적 · 2026 신규패널 한정",
    "vme_new": "VME ★우리만 낸다 (운영✓ cdn✗) — 전체",
    "vme_shared": "VME 운영도 낸다 — 라벨·이미지 한계 의심",
    "vme_op_only": "VME 운영만 낸다 — 우리가 이미 고친 것",
    "vme_all": "VME cdn_6h 전체",

    # ────────── 운영 vs 구조 모델 비교 ──────────
    "op_only_all": "운영비교 ★운영만 맞힘 — 구조 전멸(18/18)",
    "op_only_harm": "운영비교 운영✓ · 구조 ≥5 임상오류",
    "op_only_hard": "운영비교 운영✓ · 구조 과반 실패(≥9/18)",
    "op_only": "운영비교 운영✓ · 구조 일부 실패",
    "op_vs_struct": "운영비교 (참고) 구조 ≥1 실패",
    "cdn6h_worse": "운영비교 ★cdn_6h 열위 (운영✓ cdn✗)",
    "cdn6h_better": "운영비교 (참고) cdn_6h 우위 (운영✗ cdn✓)",
    "allwrong": "운영비교 전 구조 실패 웰 포함",

    # ────────── 궤적·기술오류 ──────────
    "skip_growth": "궤적 skip growth (농도 비단조 — 물리 위반)",
    "skip_ref": "궤적 ★skip 기준 사례 (사람이 고름)",
    "late_growth_miss": "궤적 ★늦게 시작 · 규칙 놓침",
    "late_growth": "궤적 늦게 시작(t5~t6 상승)",
    "te_suspect": "기술오류 TE 의심 (기포·필름)",
    "frame_disorder": "기술오류 프레임 순서 이상",

}


# ★사진 기본 표시 프레임 수 — t0~t6(6시간). 2.5 는 원래 7장이고,
#   3.0 은 10시간까지 있어 기본이 길어진다. 판정 기준이 6시간이므로 여기 맞춘다.
DEFAULT_TIME_LEN = 7

# ★드롭다운 그룹 순서 — 임상 위험도가 큰 것부터.
SUBSET_GROUPS = ["VME", "운영비교", "궤적", "기술오류", "(옛 기준)"]


def _subset_order(sid: str):
    lb = SUBSET_LABELS.get(sid, sid)
    for i, g in enumerate(SUBSET_GROUPS):
        if lb.startswith(g):
            return (i, lb)
    return (len(SUBSET_GROUPS), lb)


def load_subsets() -> dict:
    """subsets/*.csv -> {subset_id: {(project_id, sample_id, antimicrobial), ...}}"""
    out = {}
    if not os.path.isdir(SUBSET_DIR):
        return out
    for fn in sorted(os.listdir(SUBSET_DIR)):
        if not fn.endswith(".csv"):
            continue
        sid = fn[:-4]
        try:
            df = pd.read_csv(os.path.join(SUBSET_DIR, fn), dtype=str).fillna("")
        except Exception:  # noqa: BLE001 — 깨진 CSV 하나 때문에 서버가 죽지 않게 한다
            continue
        need = {"project_id", "sample_id", "antimicrobial"}
        if not need <= set(df.columns):
            continue
        out[sid] = {
            (r.project_id.strip(), r.sample_id.strip(), r.antimicrobial.strip())
            for r in df.itertuples()
        }
    return out


SUBSETS = load_subsets()


# 라벨 결과는 균종·약제 구분 없이 이 파일 하나로 관리한다.
CSV_PATH = os.path.join(HERE, "image_mic_labels.csv")
CSV_FIELDS = [
    "project_id",
    "sample_id",
    # ★같은 (project, sample, drug) 가 **두 번 시험**된 검체가 있다(접종량 차이 등, 23패널).
    #   3중 키만으로는 어느 시험에 붙은 라벨인지 구분할 수 없다(2026-09-11).
    #   기존 라벨(236건)은 이 칸이 비어 있고, `load_labels` 가 3중 키로도 읽어 유실이 없다.
    "sample_dir_id",
    "organism_group",
    "microbial_id",
    "antimicrobial",
    "image_mic",
    "image_mic_order",
    "ambiguous",
    "TE",
    # ★웰(농도) 단위 TE. 패널 TE 와 별개다 — 패널 전체가 아니라 "이 칸만" 이상할 때 쓴다.
    #   토큰: 농도 행 `c<i>`(0=최저농도) · control 행 `k<i>`(0=첫 control). 예: "k1 c0 c3"
    "TE_wells",
]


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def choice_to_order(choice: str) -> str:
    """내부 choice -> CSV image_mic_order. 'c3'->'3', 'all_growth'->'14'."""
    if choice == "all_growth":
        return "14"
    m = re.match(r"^c(\d+)$", choice)
    return m.group(1) if m else ""


def order_to_choice(order) -> str:
    """CSV image_mic_order(및 구버전 choice) -> 내부 choice."""
    v = str(order).strip()
    if v == "":
        return ""
    if v in ("all_growth", "14"):
        return "all_growth"
    if v.startswith("c"):
        return v
    if v.isdigit():
        return f"c{v}"
    return ""


os.makedirs(CELLS_DIR, exist_ok=True)
_lock = threading.Lock()


# ----------------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------------
def _fmt_num(v: float) -> str:
    """32.0 -> '32', 0.5 -> '0.5' 형태로 포맷."""
    return f"{v:g}"


def _safe_key(sample_id: str, antimicrobial: str) -> str:
    raw = f"{sample_id}__{antimicrobial}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw)


BUGMAP_COLS = ["project_id", "sample_id", "organism_group", "microbial_id"]
BMD_COLS = ["project_id", "sample_id", "antimicrobial", "bmd_mic"]


def load_allinfo_maps(src: dict) -> tuple:
    """allinfo에서 (균종 매핑, BMD MIC 매핑)을 뽑는다.

    allinfo CSV가 수백 MB라 필요한 컬럼만 한 번 읽어 캐시 폴더에 parquet 두 개로 저장한다
    (allinfo가 더 새로우면 자동 재생성).
    - 균종 매핑: sample 단위. 한 sample에 값이 여러 개면 임의로 고르지 않고
      ' / '로 이어붙여 모호함을 드러낸다.
    - BMD 매핑: sample x 약제 단위. allinfo가 적어 준 원본 표기를 그대로 쓴다
      (`POS`/`NEG`, `4/2` 같은 조합약제 표기, `>8` 등 라디오 눈금과 다른 값이 섞여 있다).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    bug_path = os.path.join(CACHE_DIR, f"{src['id']}_bugmap.parquet")
    bmd_path = os.path.join(CACHE_DIR, f"{src['id']}_bmdmic.parquet")
    stale = any(
        not os.path.exists(p) or os.path.getmtime(p) < os.path.getmtime(src["allinfo"])
        for p in (bug_path, bmd_path)
    )
    if stale:
        raw = pd.read_csv(src["allinfo"], usecols=BUGMAP_COLS + ["antimicrobial", "bmd_mic"])

        bug = raw[BUGMAP_COLS].drop_duplicates().sort_values(BUGMAP_COLS).astype(str)
        bug = bug.groupby(["project_id", "sample_id"], as_index=False).agg(
            {
                "organism_group": lambda s: " / ".join(dict.fromkeys(s)),
                "microbial_id": lambda s: " / ".join(dict.fromkeys(s)),
            }
        )
        bug.to_parquet(bug_path, index=False)

        bmd = raw[BMD_COLS].dropna(subset=["bmd_mic"]).astype(str)
        bmd = bmd.drop_duplicates(subset=["project_id", "sample_id", "antimicrobial"])
        bmd.to_parquet(bmd_path, index=False)

    return pd.read_parquet(bug_path), pd.read_parquet(bmd_path)


def resolve_path(src: dict, path: str) -> str:
    """parquet에 적힌 safetensors 경로를 이 머신의 실제 경로로 바꾼다."""
    for prefix, replacement in src.get("path_remap", ()):
        if path.startswith(prefix):
            return replacement + path[len(prefix) :]
    return path


def load_samples() -> list[dict]:
    """모든 소스의 샘플을 하나의 리스트로 로드 (idx는 전역 고유)."""
    samples = []
    used_keys = set()
    for src in SOURCES:
        df = pd.read_parquet(src["parquet"])
        if src.get("allinfo"):
            bugmap, bmdmap = load_allinfo_maps(src)
            # 균종 컬럼이 parquet 에도 있으면 merge 가 _x/_y 로 쪼개지므로, allinfo 쪽을 쓰도록 미리 버린다
            df = df.drop(columns=[c for c in ("organism_group", "microbial_id") if c in df.columns])
            df = df.merge(bugmap, on=["project_id", "sample_id"], how="left")
            df = df.merge(bmdmap, on=["project_id", "sample_id", "antimicrobial"], how="left")
        # ★allinfo 가 없는 소스(2.5 d170)는 parquet 이 이미 균종·bmd 를 들고 있다.
        for c in ("organism_group", "microbial_id"):
            if c not in df.columns:
                df[c] = "Unknown"
        for c in ("bmd_mic", "bmd_mic_order"):
            if c not in df.columns:
                df[c] = ""
        df[["organism_group", "microbial_id"]] = df[["organism_group", "microbial_id"]].fillna(
            "Unknown"
        )
        df["bmd_mic"] = df["bmd_mic"].fillna("")
        # 트리 순서대로 정렬해 두면 사이드바와 CSV 출력이 모두 보기 좋다
        df = df.sort_values(
            ["antimicrobial", "organism_group", "microbial_id", "sample_id"], kind="stable"
        )
        for row in df.to_dict("records"):
            conc_raw = str(row["concentration_list"])
            concentrations = [c.strip() for c in conc_raw.split(",") if c.strip() != ""]
            loader = src.get("loader", "safetensors")
            if loader == "png25":
                # 2.5 는 웰마다 PNG 7장이다. 경로 격자(행×시점)를 그대로 들고 간다.
                path = ""
                png_frames = json.loads(row["png_frames"])
                control_len = int(row["control_len"])
                # ★TE(기술오류) 행별 점수 [bubble, film]. 없으면 None.
                te_rows = json.loads(row["te_rows"]) if "te_rows" in row and row["te_rows"] else None
            else:
                path = str(row["image_safetensors_path"])
                png_frames = None
                te_rows = None
                m = re.search(r"_(\d+)control", os.path.basename(path))
                control_len = int(m.group(1)) if m else 1
            sample_id = str(row["sample_id"])
            antimicrobial = str(row["antimicrobial"])
            hide_bmd = antimicrobial in BMD_HIDDEN_DRUGS
            # 이미지 캐시 키: 기존 캐시를 그대로 재사용하되, 소스 간 충돌 시에만 접두사 추가
            # ★같은 (sample_id, drug) 가 **두 번 시험**된 검체가 있다(2026-09-11).
            #   캐시 키가 같으면 두 패널이 **같은 PNG 를 재사용**해 이미지가 섞인다.
            #   sample_dir_id 로 구분하고, 그래도 충돌하면 소스 접두사를 붙인다.
            _sdir = str(row.get("sample_dir_id") or "")
            key = _safe_key(sample_id, antimicrobial)
            if key in used_keys and _sdir:
                key = _safe_key(f"{sample_id}__{_sdir}", antimicrobial)
            if key in used_keys:
                key = _safe_key(f"{src['id']}__{sample_id}__{len(samples)}", antimicrobial)
            used_keys.add(key)
            samples.append(
                {
                    "idx": len(samples),
                    "src_id": src["id"],
                    "src_label": src["label"],
                    "project_id": str(row["project_id"]),
                    "sample_id": sample_id,
                    "sample_dir_id": _sdir,        # ★model_rows 4중 키 조회용
                    "organism_group": str(row["organism_group"]),
                    "microbial_id": str(row["microbial_id"]),
                    "antimicrobial": antimicrobial,
                    "group": [
                        src["label"],
                        antimicrobial,
                        str(row["organism_group"]),
                        str(row["microbial_id"]),
                    ],
                    "concentrations": concentrations,
                    "bmd_mic": "" if hide_bmd else str(row["bmd_mic"]),
                    # df의 bmd_mic_order는 image_mic_order와 같은 눈금(0=최저농도, 14=전 농도 성장)
                    "bmd_choice": "" if hide_bmd else order_to_choice(row["bmd_mic_order"]),
                    "path": path,
                    # ★캐시 무효화 기준(2026-09-14) — 격자는 이 parquet 이 정하므로
                    #   PNG 원본 mtime 이 아니라 **이 파일의 mtime** 을 봐야 갱신이 감지된다.
                    "src_path": src["parquet"],
                    "loader": loader,
                    "png_frames": png_frames,
                    "te_rows": te_rows,
                    "control_len": control_len,
                    "key": key,
                    "subsets": sorted(
                        sid for sid, keys in SUBSETS.items()
                        if (str(row["project_id"]), sample_id, antimicrobial) in keys
                    ),
                }
            )
    return samples


def source_version() -> str:
    """소스 parquet 들의 mtime 합. ★idx 는 순번이라 소스가 바뀌면 통째로 밀린다.
    열어 둔 브라우저가 옛 idx 로 저장하면 **다른 패널에 라벨이 붙는다**(2026-09-08 실제 발생).
    클라이언트가 이 값을 들고 있다가 다르면 저장을 거부하고 새로고침을 요구한다."""
    v = []
    for src in SOURCES:
        try:
            v.append(str(int(os.path.getmtime(src["parquet"]))))
        except OSError:
            v.append("0")
    return "-".join(v)


# ★모델 판정(행별) — 소스 parquet 을 건드리지 않으려고 **별도 파일**로 붙인다.
#   parquet 을 바꾸면 idx 가 밀려 열어 둔 탭이 깨진다(2026-09-08 사고).
#   생성 = 예측 parquet → model_rows.json. 없으면 배지 없이 그대로 뜬다.
MODEL_ROWS = {}
try:
    with open(os.path.join(HERE, "model_rows.json"), encoding="utf-8") as _f:
        MODEL_ROWS = json.load(_f)
except (OSError, ValueError):
    MODEL_ROWS = {}


SOURCE_VERSION = source_version()
SAMPLES = load_samples()
SAMPLES_BY_IDX = {s["idx"]: s for s in SAMPLES}

# (sample_id, antimicrobial) -> 샘플들. 주소로 바로 여는 /api/resolve 용 (대소문자 무시).
SAMPLES_BY_NAME: dict = {}
for _s in SAMPLES:
    SAMPLES_BY_NAME.setdefault(
        (_s["sample_id"].lower(), _s["antimicrobial"].lower()), []
    ).append(_s)


def resolve_sample(sample_id: str, antimicrobial: str, project_id: str = "") -> dict:
    """sample_id / antimicrobial (+ 선택적 project_id)로 샘플을 찾는다.

    project_id를 생략해도 후보가 하나뿐이면 그대로 연다.
    """
    hits = SAMPLES_BY_NAME.get((sample_id.strip().lower(), antimicrobial.strip().lower()), [])
    pid = project_id.strip().lower()
    if pid:
        hits = [s for s in hits if s["project_id"].lower() == pid]
    if len(hits) == 1:
        return {"ok": True, "idx": hits[0]["idx"]}
    if not hits:
        return {"ok": False, "error": "해당 샘플을 찾지 못했습니다."}
    return {
        "ok": False,
        "error": "project_id가 필요합니다 (후보 여러 개).",
        "candidates": [
            {"project_id": s["project_id"], "idx": s["idx"], "group": s["group"]} for s in hits
        ],
    }


# ----------------------------------------------------------------------------
# 라벨(CSV) 저장소
# ----------------------------------------------------------------------------
def label_key(sample: dict) -> tuple:
    """★4중 키(2026-09-11). `sample_dir_id` 가 없으면 3중과 같아져 구본과 호환된다."""
    return (sample["project_id"], sample["sample_id"], sample["antimicrobial"],
            str(sample.get("sample_dir_id") or ""))


def load_labels() -> dict:
    """단일 CSV에서 라벨을 읽는다."""
    labels = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sd = str(r.get("sample_dir_id", "") or "")
                key = (r.get("project_id", ""), r.get("sample_id", ""),
                       r.get("antimicrobial", ""), sd)
                labels[key] = {
                    "human_mic": r.get("image_mic", r.get("human_mic", "")),
                    "choice": order_to_choice(r.get("image_mic_order", r.get("choice", ""))),
                    "ambiguous": _truthy(r.get("ambiguous", "")),
                    "TE": _truthy(r.get("TE", "")),
                    "TE_wells": [t for t in str(r.get("TE_wells", "")).split() if t],
                }
    return labels


LABELS = load_labels()


def label_get(sample: dict):
    """★4중 키 우선, 없으면 **구본 3중 키**(sample_dir_id="")로 폴백(2026-09-11).

    기존 라벨 236건은 `sample_dir_id` 칸이 비어 있다. 폴백이 없으면 전부 사라져 보인다.
    ⚠두 번 시험된 패널(23개)에서는 구본 라벨이 **양쪽 모두에 보인다** — 어느 시험에
      붙었는지 원본에 없기 때문이다. 새로 저장하면 4중 키로 확정된다.
    """
    k = label_key(sample)
    v = LABELS.get(k)
    if v is None and k[3]:
        v = LABELS.get((k[0], k[1], k[2], ""))
    return v


def save_labels():
    """전체 라벨을 단일 CSV로 원자적 저장 (샘플 정렬 순서 유지)."""
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        seen = set()
        for s in SAMPLES:
            key = label_key(s)
            lab = LABELS.get(key)
            if lab is None and key[3]:              # ★구본 3중 키 폴백
                key = (key[0], key[1], key[2], "")
                lab = LABELS.get(key)
            if lab is not None and key not in seen:
                seen.add(key)
                w.writerow(
                    {
                        "project_id": s["project_id"],
                        "sample_id": s["sample_id"],
                        "sample_dir_id": str(s.get("sample_dir_id") or ""),
                        "organism_group": s["organism_group"],
                        "microbial_id": s["microbial_id"],
                        "antimicrobial": s["antimicrobial"],
                        "image_mic": lab["human_mic"],
                        "image_mic_order": choice_to_order(lab["choice"]),
                        "ambiguous": "1" if lab.get("ambiguous") else "0",
                        "TE": "1" if lab.get("TE") else "0",
                        "TE_wells": " ".join(lab.get("TE_wells") or []),
                    }
                )
    os.replace(tmp, CSV_PATH)


def get_entry(sample: dict) -> dict:
    """샘플의 라벨 항목을 반환 (없으면 기본값으로 새로 생성).

    ★구본 3중 키 항목이 있으면 **4중 키로 이관**한다(2026-09-11). 안 하면 편집 순간
      빈 항목이 덮어써져 기존 라벨이 사라진다.
    """
    key = label_key(sample)
    if key not in LABELS:
        old = LABELS.pop((key[0], key[1], key[2], ""), None) if key[3] else None
        LABELS[key] = old if old is not None else {
            "human_mic": "", "choice": "", "ambiguous": False, "TE": False, "TE_wells": []}
    return LABELS[key]


def count_done() -> int:
    """MIC(choice)가 지정된 샘플 수. 플래그만 있는 항목은 미완료로 간주."""
    done = 0
    for s in SAMPLES:
        lab = label_get(s)
        if lab and lab.get("choice"):
            done += 1
    return done


# ----------------------------------------------------------------------------
# 셀 이미지 추출 (lazy, 캐시)
# ----------------------------------------------------------------------------
_cell_locks: dict = {}
_cell_locks_guard = threading.Lock()


def _cell_lock(key: str) -> threading.Lock:
    """샘플별 추출 락. 프리페치와 사용자 요청이 같은 샘플을 중복 추출하지 않도록."""
    with _cell_locks_guard:
        return _cell_locks.setdefault(key, threading.Lock())


def _cache_stale(sample: dict, done_flag: str) -> bool:
    """원본이 캐시보다 새로우면(데이터 갱신) 캐시를 버린다.

    ★**소스 parquet 의 mtime 도 본다**(2026-09-14 수정). 예전엔 2.5(png25)에서
      **첫 프레임 PNG** 의 mtime 만 봤는데, 그것은 NAS 원본이라 **우리가 격자를 고쳐도
      절대 변하지 않는다.** 그래서 프레임 순서를 정정해도 캐시가 그대로 재사용됐다
      — 실측 사고: 정본 전환(09-11 17:11) 1시간 36분 **전**에 만들어진 캐시
      (`E187S06R0__CIP` 09-11 15:35)가 계속 서빙돼 라벨러가 **뒤섞인 순서**를 보고 있었다.
      격자는 소스 parquet 이 정하므로 그 파일의 mtime 이 진짜 기준이다.
    """
    try:
        t_done = os.path.getmtime(done_flag)
        src = sample.get("src_path")
        if src and os.path.exists(src) and t_done < os.path.getmtime(src):
            return True                       # ★소스 갱신 → 무조건 다시 뽑는다
        if sample.get("loader") == "png25":
            grid = sample.get("png_frames") or []
            first = next((c for r in grid for c in r if c), None)
            if not first:
                return False
            return t_done < os.path.getmtime(first)
        return t_done < os.path.getmtime(sample["path"])
    except OSError:
        # 원본이 없으면 이미 뽑아 둔 캐시라도 그대로 쓴다
        return False


def is_cached(sample: dict) -> bool:
    done_flag = os.path.join(CELLS_DIR, sample["key"], ".done")
    return os.path.exists(done_flag) and not _cache_stale(sample, done_flag)


def ensure_cells(sample: dict) -> dict:
    """safetensors -> 셀 PNG. (row, col) 그리드 레이아웃 반환."""
    n_conc = len(sample["concentrations"])
    control_len = sample["control_len"]
    n_rows = n_conc + control_len

    out_dir = os.path.join(CELLS_DIR, sample["key"])
    done_flag = os.path.join(out_dir, ".done")

    with _cell_lock(sample["key"]):
        _extract_cells(sample, out_dir, done_flag, n_rows)

    with open(done_flag) as f:
        meta = json.load(f)
    time_len = meta["time_len"]
    # 캐시 버스터: 재추출하면 .done 의 mtime 이 바뀌므로 URL 도 같이 바뀐다.
    # (URL 이 고정이면 브라우저가 max-age 동안 예전 PNG 를 계속 쓴다)
    ver = int(os.path.getmtime(done_flag))

    rows = []
    for row in range(n_rows):
        is_control = row < control_len
        conc_index = None if is_control else row - control_len
        label = "Cont" if is_control else sample["concentrations"][conc_index]
        cells = [
            f"/static/cells/{sample['key']}/r{row}_t{col}.png?v={ver}" for col in range(time_len)
        ]
        # 모델 판정 — 농도 행에만 붙는다(control 은 모델 대상이 아니다)
        mj = None
        # ★4중 키 우선(2026-09-11) — 같은 (project,sample,drug)가 두 번 시험된 검체가 있어
        #   3중 키면 한쪽 판정이 다른 쪽에 잘못 붙는다. 구본 파일 호환으로 3중 키 폴백.
        _k3 = f'{sample["project_id"]}|{sample["sample_id"]}|{sample["antimicrobial"]}'
        mr = (MODEL_ROWS.get(f'{_k3}|{sample.get("sample_dir_id", "")}')
              or MODEL_ROWS.get(_k3))
        if mr and not is_control and conc_index is not None:
            def _g(seq):
                if not seq or conc_index >= len(seq):
                    return None
                v = seq[conc_index]
                return None if v == "" else ("G" if v == "0" else "NG")
            mj = {"cdn": _g(mr.get("cdn")), "op": _g(mr.get("op")),
                  "nG": (mr.get("nG") or [None])[conc_index]
                        if mr.get("nG") and conc_index < len(mr["nG"]) else None,
                  "nstruct": mr.get("nstruct")}
        te = None
        tr = sample.get("te_rows")
        if tr and row < len(tr) and tr[row]:
            v = tr[row]
            te = {"bubble": v[0], "film": v[1],
                  "collapse": bool(v[2]) if len(v) > 2 else False,
                  "drop": v[3] if len(v) > 3 else 0}
        rows.append(
            {"label": label, "is_control": is_control, "conc_index": conc_index,
             "cells": cells, "te": te, "mj": mj}
        )

    _mr = MODEL_ROWS.get(
        f'{sample["project_id"]}|{sample["sample_id"]}|{sample["antimicrobial"]}')
    # ★기본 표시는 t0~t6(7프레임)까지만 — 판정이 6시간 기준이라 그 이후는 참고다.
    #   전체(3.0 은 10시간까지)는 UI 의 "10h" 옵션으로 켠다. 자르는 것은 프론트가 하고,
    #   여기서는 전체를 주되 기본 길이를 함께 알려 준다(2026-09-14 사용자 지시).
    return {"time_len": time_len, "time_len_default": min(time_len, DEFAULT_TIME_LEN),
            "rows": rows,
            "cdn_mic": (_mr or {}).get("cdn_mic"), "op_mic": (_mr or {}).get("op_mic")}


def cached_cell_urls(sample: dict) -> list:
    """이미 캐시된 샘플의 셀 URL 목록. **추출은 하지 않는다.**

    브라우저가 다음 샘플을 미리 받아 두게 하려고 /api/sample 응답에 실어 보낸다.
    아직 캐시가 없으면 빈 리스트를 준다 — 프리페치가 수백 ms 짜리 추출을 기다리며
    브라우저 연결 슬롯을 잡고 있으면 정작 지금 보는 화면이 느려진다.
    """
    done_flag = os.path.join(CELLS_DIR, sample["key"], ".done")
    try:
        ver = int(os.path.getmtime(done_flag))
        with open(done_flag) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return []
    if _cache_stale(sample, done_flag):
        return []
    return [
        f"/static/cells/{sample['key']}/r{row}_t{col}.png?v={ver}"
        for row in range(meta["n_rows"])
        for col in range(meta["time_len"])
    ]


def _extract_cells(sample: dict, out_dir: str, done_flag: str, n_rows: int):
    if os.path.exists(done_flag) and _cache_stale(sample, done_flag):
        # 그리드 크기가 바뀌었을 수 있으니 이전 PNG 를 지우고 통째로 다시 뽑는다
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
    if not os.path.exists(done_flag):
        os.makedirs(out_dir, exist_ok=True)
        if sample.get("loader") == "png25":
            # ★2.5 — 웰마다 PNG 가 이미 디스크에 있다. 잘라낼 게 없어 **그대로 복사**한다.
            #   원본이 224x119 비정방형이라 리사이즈하지 않는다(왜곡 금지).
            #   경로가 비면(구형 레이아웃 등) 검은 칸을 채워 격자를 유지한다.
            grid = sample["png_frames"]
            time_len = max((len(r) for r in grid), default=0)
            for row, fr in enumerate(grid):
                for col in range(time_len):
                    dst = os.path.join(out_dir, f"r{row}_t{col}.png")
                    src_p = fr[col] if col < len(fr) else None
                    if src_p and os.path.exists(src_p):
                        try:
                            Image.open(src_p).convert("L").save(dst)
                            continue
                        except Exception:  # noqa: BLE001 — 깨진 파일은 빈 칸으로
                            pass
                    Image.new("L", (224, 119), 0).save(dst)
        else:
            if not os.path.exists(sample["path"]):
                raise FileNotFoundError(f"safetensors 없음: {sample['path']}")
            with safe_open(sample["path"], "numpy") as f:
                arr = f.get_tensor("image")  # (N, 1, H, W) uint8
            total = arr.shape[0]
            time_len = total // n_rows
            for idx in range(total):
                row = idx // time_len
                col = idx % time_len
                cell = arr[idx, 0]
                Image.fromarray(cell, "L").save(os.path.join(out_dir, f"r{row}_t{col}.png"))
        # .done 은 마지막에 써서, 중간에 죽으면 다음 실행이 다시 추출하도록 한다
        with open(done_flag, "w") as f:
            f.write(json.dumps({"time_len": time_len, "n_rows": n_rows}))


# ----------------------------------------------------------------------------
# 프리페치 — 다음에 볼 샘플을 미리 추출해 둔다
# ----------------------------------------------------------------------------
PREFETCH_AHEAD = int(os.environ.get("PREFETCH_AHEAD", "8"))
PREFETCH_WORKERS = int(os.environ.get("PREFETCH_WORKERS", "2"))
# 브라우저에게 "미리 받아 두라"고 URL 을 넘겨 줄 샘플 수. 서버측 추출 선행(PREFETCH_AHEAD)
# 과 달리 실제 네트워크 전송이 일어나므로 훨씬 작게 잡는다 (샘플당 1.4~1.8MB).
PREFETCH_WARM = int(os.environ.get("PREFETCH_WARM", "2"))

_prefetch_q: "queue.Queue[int]" = queue.Queue()
_prefetch_queued: set = set()
_prefetch_guard = threading.Lock()


def _prefetch_worker():
    while True:
        idx = _prefetch_q.get()
        try:
            s = SAMPLES_BY_IDX.get(idx)
            if s and not is_cached(s):
                ensure_cells(s)
        except Exception:  # noqa: BLE001 — 프리페치 실패는 조용히 무시 (열 때 다시 시도)
            pass
        finally:
            with _prefetch_guard:
                _prefetch_queued.discard(idx)
            _prefetch_q.task_done()


def prefetch_idxs(idxs, ahead: int = PREFETCH_AHEAD):
    """주어진 idx들을 앞에서부터 최대 ahead 개까지 큐에 넣는다."""
    for j in list(idxs)[:ahead]:
        s = SAMPLES_BY_IDX.get(j)
        if not s or is_cached(s):
            continue
        with _prefetch_guard:
            if j in _prefetch_queued:
                continue
            _prefetch_queued.add(j)
        _prefetch_q.put(j)


def schedule_prefetch(idx: int, ahead: int = PREFETCH_AHEAD):
    """idx 다음 샘플들을 큐에 넣는다 (서버 기본 정렬 순서).

    사이드바 그룹핑 순서를 바꾸면 '다음 샘플'이 idx+1이 아니게 되므로,
    브라우저가 실제 다음 순서를 보내 주면 그쪽(prefetch_idxs)을 우선한다.
    """
    prefetch_idxs(range(idx + 1, min(idx + 1 + ahead, len(SAMPLES))), ahead)


def start_prefetch_workers():
    for _ in range(PREFETCH_WORKERS):
        threading.Thread(target=_prefetch_worker, daemon=True).start()


def choice_to_mic(sample: dict, choice: str) -> str:
    """라디오 choice -> 휴먼 MIC 문자열.

    'c{j}'      : j번째 농도 (j==0 이면 '<=최저농도')
    'all_growth': 최고농도까지 성장 -> '>=(최고농도*2)'
    """
    concentrations = sample["concentrations"]
    if choice == "all_growth":
        hi = 2.0 * float(concentrations[-1])
        return f">={_fmt_num(hi)}"
    m = re.match(r"^c(\d+)$", choice)
    if not m:
        raise ValueError(f"잘못된 choice: {choice}")
    j = int(m.group(1))
    if j < 0 or j >= len(concentrations):
        raise ValueError(f"농도 인덱스 범위 초과: {j}")
    if j == 0:
        return f"<={concentrations[0]}"
    return concentrations[j]


# ----------------------------------------------------------------------------
# HTTP 핸들러
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    # 기본값은 HTTP/1.0 이라 keep-alive 가 꺼진다. 샘플 하나가 셀 이미지 49~70장이므로
    # 그만큼 TCP 연결을 새로 맺고 끊게 된다. (_json/_bytes 모두 Content-Length 를
    # 정확히 보내고 있으므로 1.1 로 올려도 안전하다.)
    protocol_version = "HTTP/1.1"
    # keep-alive 를 켜면 유휴 연결이 readline() 에서 스레드를 무한정 붙잡는다.
    # 브라우저 하나가 6연결을 쓰므로 타임아웃이 없으면 스레드가 쌓이기만 한다.
    timeout = 10

    def log_message(self, *args):
        pass  # 조용히

    # ---- 응답 헬퍼 ----
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        # /api/samples 는 38,000건이 넘어 12MB 가 나온다. gzip 이면 40배 줄어든다.
        # 작은 응답(라벨 저장 등)은 압축이 오히려 손해라 임계값을 둔다.
        encoding = None
        if len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, 6)
            encoding = "gzip"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body, content_type, status=200, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            # URL 에 ?v=<mtime> 이 붙어 있으므로 오래 캐시해도 안전하다
            self.send_header("Cache-Control", IMG_CACHE_CONTROL)
        else:
            # index.html / JSON 은 절대 캐시하지 않는다 (예전 화면·데이터 방지)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def _sample_status(self, s):
        lab = label_get(s)
        return {
            "idx": s["idx"],
            # group 은 서버 기본 순서. 사이드바는 아래 필드로 순서를 바꿔 가며 직접 조립한다.
            "group": s["group"],
            "src_label": s["src_label"],
            "sample_id": s["sample_id"],
            "organism_group": s["organism_group"],
            "microbial_id": s["microbial_id"],
            "antimicrobial": s["antimicrobial"],
            "labeled": bool(lab and lab.get("choice")),
            "human_mic": lab["human_mic"] if lab else "",
            "choice": lab["choice"] if lab else "",
            "ambiguous": bool(lab and lab.get("ambiguous")),
            "TE": bool(lab and lab.get("TE")),
            "TE_wells": (lab.get("TE_wells") if lab else []) or [],
            # 이 샘플이 속한 서브셋들. 사이드바 "이것만 보기" 필터가 쓴다.
            "subsets": s["subsets"],
            # ★bmd 유무(2026-09-14) — 사이드바 "bmd 있는 것만" 체크박스가 쓴다.
            #   소스 6,020 중 2,483(41%)이 bmd 결측이고 **프로젝트 단위로 갈린다**
            #   (202408_KNUH·HEGP_2022 0% · KUMC_2025 6% vs SNUH_2019 99.5%).
            #   bmd 가 없으면 화면의 `bmd · 운영 · 구조모델` 세 칸이 다 비어 교차검증이 안 된다.
            #   ⚠소스에서 빼지는 않는다 — 정답이 없는 검체라 사람 라벨의 가치는 오히려 크다.
            "has_bmd": bool(str(s.get("bmd_mic") or "").strip()),
        }

    # ---- GET ----
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            with open(INDEX_HTML, "rb") as f:
                self._bytes(f.read(), "text/html; charset=utf-8")
            return

        if path == "/api/subsets":
            self._json({
                "subsets": [
                    {"id": sid,
                     "label": SUBSET_LABELS.get(sid, sid),
                     "count": sum(1 for x in SAMPLES if sid in x["subsets"])}
                    # ★그룹 → 라벨 순 정렬. 접두어(VME/운영비교/궤적/기술오류)로 묶여 보이고
                    #   "(옛 기준)" 은 맨 아래로 간다.
                    #   건수는 UI 가 붙이므로 라벨에 적지 않는다(옛 수치가 남는 사고를 막는다).
                    for sid in sorted(SUBSETS, key=_subset_order)
                ]
            })
            return

        if path == "/api/samples":
            total = len(SAMPLES)
            done = count_done()
            self._json(
                {
                    "version": SOURCE_VERSION,
                    "total": total,
                    "done": done,
                    "samples": [self._sample_status(s) for s in SAMPLES],
                }
            )
            return

        if path == "/api/resolve":
            self._json(
                resolve_sample(
                    qs.get("sample_id", [""])[0],
                    qs.get("antimicrobial", [""])[0],
                    qs.get("project_id", [""])[0],
                )
            )
            return

        if path == "/api/sample":
            try:
                idx = int(qs.get("idx", ["-1"])[0])
                s = SAMPLES_BY_IDX[idx]
            except (ValueError, KeyError):
                self._json({"error": "invalid idx"}, 400)
                return
            lab = label_get(s)
            # 다음 샘플들을 백그라운드에서 미리 추출.
            # next=1,2,3 이 오면 사이드바에 실제로 보이는 다음 순서를 쓴다.
            nxt = [int(t) for t in qs.get("next", [""])[0].split(",") if t.strip().isdigit()]
            if nxt:
                prefetch_idxs(nxt)
            else:
                schedule_prefetch(idx)
                nxt = list(range(idx + 1, min(idx + 1 + PREFETCH_AHEAD, len(SAMPLES))))
            # 브라우저가 다음 샘플 이미지를 미리 받아 둘 수 있게 URL 을 실어 보낸다.
            # 이미 캐시된 것만 넣으므로 추가 왕복도, 추출 대기도 없다.
            warm = []
            for j in nxt[:PREFETCH_WARM]:
                s_next = SAMPLES_BY_IDX.get(j)
                if s_next:
                    warm.extend(cached_cell_urls(s_next))
            try:
                layout = ensure_cells(s)
            except Exception as e:  # noqa: BLE001
                self._json(
                    {
                        "idx": idx,
                        "project_id": s["project_id"],
                        "sample_id": s["sample_id"],
                        # ★pkey — 클라이언트가 저장 시 되돌려 보내 stale idx 를 막는다.
                        #   4중(sample_dir_id 포함) — 두 번 시험된 검체를 구분한다.
                        "pkey": (f'{s["project_id"]}|{s["sample_id"]}|{s["antimicrobial"]}'
                                 f'|{s.get("sample_dir_id") or ""}'),
                        "microbial_id": s["microbial_id"],
                        "antimicrobial": s["antimicrobial"],
                        "bmd_mic": s["bmd_mic"],
                        "bmd_choice": s["bmd_choice"],
                        "ambiguous": bool(lab and lab.get("ambiguous")),
                        "TE": bool(lab and lab.get("TE")),
            "TE_wells": (lab.get("TE_wells") if lab else []) or [],
                        "error": str(e),
                    },
                    200,
                )
                return
            hi = _fmt_num(2.0 * float(s["concentrations"][-1]))
            self._json(
                {
                    "idx": idx,
                    "project_id": s["project_id"],
                    "sample_id": s["sample_id"],
                    # ★pkey — 저장 시 되돌려 보내 stale idx 를 막는다(4중: 두 번 시험 구분)
                    "pkey": (f'{s["project_id"]}|{s["sample_id"]}|{s["antimicrobial"]}'
                             f'|{s.get("sample_dir_id") or ""}'),
                    "microbial_id": s["microbial_id"],
                    "antimicrobial": s["antimicrobial"],
                    "concentrations": s["concentrations"],
                    "time_len": layout["time_len"],
                    # ★기본 표시 범위(t0~t6). 프론트가 이걸로 자른다.
                    "time_len_default": layout.get("time_len_default"),
                    "rows": layout["rows"],
                    # 모델 MIC — 사람 판독과 대조용(참고 표시)
                    "op_mic": layout.get("op_mic"),
                    "cdn_mic": layout.get("cdn_mic"),
                    "all_growth_label": f">={hi}",
                    "bmd_mic": s["bmd_mic"],
                    "bmd_choice": s["bmd_choice"],
                    "human_mic": lab["human_mic"] if lab else "",
                    "choice": lab["choice"] if lab else "",
                    "ambiguous": bool(lab and lab.get("ambiguous")),
                    "TE": bool(lab and lab.get("TE")),
            "TE_wells": (lab.get("TE_wells") if lab else []) or [],
                    "prefetch": warm,
                }
            )
            return

        if path.startswith("/static/cells/"):
            rel = path[len("/static/cells/") :]
            rel = urllib.parse.unquote(rel)
            fp = os.path.normpath(os.path.join(CELLS_DIR, rel))
            if not fp.startswith(CELLS_DIR + os.sep) or not os.path.isfile(fp):
                self._bytes(b"", "image/png", 404)
                return
            st = os.stat(fp)
            etag = f'"{int(st.st_mtime)}-{st.st_size}"'
            # 브라우저가 재검증할 때(Ctrl+R 등) 본문 전송을 통째로 없앤다.
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", IMG_CACHE_CONTROL)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(st.st_size))
            self.send_header("Cache-Control", IMG_CACHE_CONTROL)
            self.send_header("ETag", etag)
            self.end_headers()
            # 통째로 read() 하지 않고 흘려보낸다 (GIL 점유·메모리 할당 감소)
            with open(fp, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
            return

        self._bytes(b"Not Found", "text/plain", 404)

    # ---- POST ----
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        # keep-alive 에서는 본문을 끝까지 읽지 않으면 남은 바이트가 다음 요청의 시작으로
        # 해석돼 연결이 어긋난다. 경로를 따지기 전에 먼저 비워 둔다.
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        if parsed.path not in ("/api/label", "/api/flags", "/api/unlabel"):
            self._json({"error": "not found"}, 404)
            return

        try:
            data = json.loads(raw or b"{}")
            idx = int(data["idx"])
            s = SAMPLES_BY_IDX[idx]
        except (ValueError, KeyError, json.JSONDecodeError):
            self._json({"error": "invalid payload"}, 400)
            return

        # ★idx 는 소스 순번이라 소스를 다시 만들면 통째로 밀린다.
        #   열어 둔 탭이 옛 idx 로 저장하면 **다른 패널에 라벨이 붙는다**(2026-09-08 실제 발생).
        #   그래서 클라이언트가 보낸 패널 신원(pkey)·소스 버전을 대조하고, 어긋나면 거부한다.
        # ★pkey 도 4중(2026-09-11) — 두 번 시험된 검체(23패널)는 3중으로 구분되지 않아
        #   가드가 뚫린다. 구본 클라이언트(3중 pkey)는 앞 3칸만 비교해 호환한다.
        pkey = data.get("pkey")
        want = (f'{s["project_id"]}|{s["sample_id"]}|{s["antimicrobial"]}'
                f'|{s.get("sample_dir_id") or ""}')
        if pkey and pkey.count("|") == 2:                 # 구본 3중 pkey
            want = "|".join(want.split("|")[:3])
        if pkey and pkey != want:
            self._json({"error": "stale",
                        "message": "소스가 바뀌어 순번이 밀렸습니다. 새로고침 후 다시 라벨하세요.",
                        "expected": want, "got": pkey}, 409)
            return
        ver = data.get("version")
        if ver and ver != SOURCE_VERSION:
            self._json({"error": "stale",
                        "message": "소스가 갱신됐습니다. 새로고침 후 다시 라벨하세요."}, 409)
            return

        if parsed.path == "/api/flags":
            self._handle_flags(idx, s, data)
            return

        if parsed.path == "/api/unlabel":
            self._handle_unlabel(idx, s)
            return

        # /api/label — MIC 라디오 선택
        try:
            choice = str(data["choice"])
            mic = choice_to_mic(s, choice)
        except (KeyError, ValueError) as e:
            self._json({"error": str(e)}, 400)
            return

        with _lock:
            entry = get_entry(s)
            entry["human_mic"] = mic
            entry["choice"] = choice
            save_labels()
            done = count_done()

        self._json({"ok": True, "idx": idx, "human_mic": mic, "choice": choice, "done": done})

    def _handle_unlabel(self, idx, s):
        """저장된 MIC를 지운다. ambiguous/TE 플래그도 없으면 항목 자체를 제거."""
        key = label_key(s)
        with _lock:
            entry = get_entry(s)
            entry["human_mic"] = ""
            entry["choice"] = ""
            if not entry["ambiguous"] and not entry["TE"]:
                del LABELS[key]
            save_labels()
            done = count_done()

        self._json({"ok": True, "idx": idx, "human_mic": "", "choice": "", "done": done})

    def _handle_flags(self, idx, s, data):
        """ambiguous / TE 체크박스 갱신. 플래그·MIC 모두 없으면 항목 제거."""
        key = label_key(s)
        with _lock:
            entry = get_entry(s)
            if "ambiguous" in data:
                entry["ambiguous"] = bool(data["ambiguous"])
            if "TE" in data:
                entry["TE"] = bool(data["TE"])
            # ★웰 단위 TE — {"te_well": "c0", "on": true} 로 한 칸씩 토글한다.
            if "te_well" in data:
                w = str(data["te_well"])
                cur = set(entry.get("TE_wells") or [])
                cur.add(w) if data.get("on") else cur.discard(w)
                entry["TE_wells"] = sorted(cur, key=lambda t: (t[0], int(t[1:] or 0)))
            amb, te = entry["ambiguous"], entry["TE"]
            wells = entry.get("TE_wells") or []
            if not entry["choice"] and not amb and not te and not wells:
                del LABELS[key]
            save_labels()
            done = count_done()

        self._json({"ok": True, "idx": idx, "ambiguous": amb, "TE": te,
                    "TE_wells": wells, "done": done})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # 샘플 하나를 열면 셀 이미지 49~70개가 한꺼번에 몰린다. 기본값 5 는 너무 작다.
    request_queue_size = 128

    def handle_error(self, request, client_address):
        # 브라우저가 프리페치나 화면 전환 중에 전송 중인 요청을 끊는 건 정상이다.
        # 기본 구현은 이걸 전부 스택트레이스로 찍어서 콘솔을 덮어 버린다.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main():
    for src in SOURCES:
        n = sum(1 for s in SAMPLES if s["src_id"] == src["id"])
        done = sum(
            1
            for s in SAMPLES
            if s["src_id"] == src["id"] and (label_get(s) or {}).get("choice")
        )
        print(f"[human-labeling] {src['label']}: {done}/{n}")
    print(f"[human-labeling] samples: {len(SAMPLES)}  labeled: {count_done()}")
    print(f"[human-labeling] CSV: {CSV_PATH}")
    print(f"[human-labeling] prefetch: {PREFETCH_AHEAD} ahead / {PREFETCH_WORKERS} workers")
    print(f"[human-labeling] http://localhost:{PORT}")
    start_prefetch_workers()
    Server(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
