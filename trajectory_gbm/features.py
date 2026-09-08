"""Self-contained trajectory feature 추출 (signal-agnostic).

콤마 구분 시계열 컬럼(brightness / object_area 등)에서 per-(sample, drug, conc)
trajectory 를 뽑아 LightGBM 입력 feature 로 변환한다. control well = sample 별
`Cont` 행 중 CTL_PICK_INDEX 시점 값이 가장 큰 trajectory.

* 시계열 길이는 MODEL_TIME_LEN(=7) 으로 truncate (FDA 는 더 길어도 앞 7개 사용).
* feature 는 절대 trajectory(drug/ctl) + control 상대(ratio/diff) 혼합.
* feature 이름 prefix 는 `oa_` (object-area-style) 로 통일 — 어떤 신호든 동일.

원래 brightness_asym_corrector/brightness_features.py 에서 일반화. 입력 컬럼만 바꾸면
brightness·object_area 모두 동일 코드로 처리된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_TIME_LEN = 7
CTL_PICK_INDEX = 6
EPS = 1e-6
FEATURE_PREFIX = "oa_"


def parse_csv_list(s) -> np.ndarray | None:
    """콤마 구분 시계열 문자열 → float array (nan 보존)."""
    if not isinstance(s, str):
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(np.nan if x.lower() == "nan" else _to_float(x))
    return np.array(out, dtype=np.float32) if out else None


def _to_float(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return np.nan


def features_for_row(sid: str, drug: str, rank: int,
                     traj: np.ndarray, ctl: np.ndarray) -> dict:
    """약제 trajectory + control 로 hand-crafted feature 생성."""
    ratio = traj / (ctl + EPS)
    diff = traj - ctl
    feat: dict = {"sample_id": sid, "antimicrobial": drug, "concentration_idx_0": rank}
    for t in range(len(traj)):
        feat[f"oa_drug_t{t}"] = float(traj[t])
        feat[f"oa_ctl_t{t}"] = float(ctl[t])
        feat[f"oa_ratio_t{t}"] = float(ratio[t])
        feat[f"oa_diff_t{t}"] = float(diff[t])
    feat["oa_drug_auc"] = float(traj.sum())
    feat["oa_ctl_auc"] = float(ctl.sum())
    feat["oa_drug_slope"] = float(traj[-1] - traj[0])
    feat["oa_ctl_slope"] = float(ctl[-1] - ctl[0])
    feat["oa_drug_max"] = float(traj.max())
    feat["oa_drug_min"] = float(traj.min())
    feat["oa_drug_range"] = float(traj.max() - traj.min())
    feat["oa_ratio_t6"] = float(ratio[-1])
    feat["oa_ratio_mean"] = float(np.mean(ratio))
    feat["oa_diff_t6"] = float(diff[-1])
    feat["oa_auc_ratio"] = float(traj.sum() / (ctl.sum() + EPS))
    feat["oa_max_ratio"] = float(traj.max() / (ctl.max() + EPS))
    half = 0.5 * traj.max()
    feat["oa_drug_onset"] = float(np.argmax(traj >= half) if (traj >= half).any()
                                  else MODEL_TIME_LEN)
    feat["oa_drug_late_slope"] = float(traj[-1] - traj[3])
    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if "concentration_idx_0" in df.columns:
        cols.append("concentration_idx_0")
    return cols


def build_features(csv_path: str, traj_col: str,
                   label_col: str | None = None,
                   extra_cols: tuple[str, ...] = (),
                   ctl_pick: str = "t6",
                   time_idx: "tuple[int, ...] | None" = None) -> pd.DataFrame:
    """CSV → per-(sample, drug, conc) trajectory feature table.

    traj_col: 원본 trajectory 컬럼 (예: 'brightness_list' / 'object_area' / 'brightness').
    label_col: 학습 시 'bmd_gng' (G/NG). extra_cols: 예) ('train_or_test',).
    ctl_pick: control 웰 선택 규칙 — **기본 `t6`(기존 동작 유지, 3.0 영향 없음)**.
        · `t6`        : `CTL_PICK_INDEX`(=6) 시점 값이 최대인 웰
        · `maxgrowth` : **peak − t0** 가 최대인 웰(dRAST 2.5 규약, GF rule 과 동일 정의)
    time_idx: 사용할 시점 인덱스(기본 None=`range(MODEL_TIME_LEN)`, 즉 t0..t6 기존 동작).
        ★시점 수가 다른 trajectory(예: SSL 재생성 object_area 는 6h 4프레임 [0,2,4,6])를
          기존 7프레임 신호와 **같은 시각에 맞춰** 쓰기 위한 것.
          예) SSL area(len 4) → (0,1,2,3) · 원본 count(len 7) → (0,2,4,6).
          생성되는 feature 이름은 항상 t0.. 로 재번호된다(정렬 목적).
          ★2.5 실측: control 웰의 **57.9%** 가 중간에 peak 을 찍고 t6 에 내려앉는다
            (기포·초점). t6 만 보면 성장을 중앙 16.9% 과소평가해 **다른 웰을 고른다**.
    """
    assert ctl_pick in ("t6", "maxgrowth"), ctl_pick
    tix = list(range(MODEL_TIME_LEN)) if time_idx is None else list(time_idx)
    need = max(tix) + 1
    use = {"sample_id", "antimicrobial", "concentration", traj_col}
    if label_col:
        use.add(label_col)
    use |= set(extra_cols)
    df = pd.read_csv(csv_path, usecols=lambda c: c in use, low_memory=False)
    df["_tr"] = df[traj_col].apply(parse_csv_list)
    df = df.dropna(subset=["_tr"]).reset_index(drop=True)

    # control trajectory per sample
    ctl_map: dict[str, np.ndarray] = {}
    for sid, sub in df[df["antimicrobial"] == "Cont"].groupby("sample_id", sort=False):
        max_len = int(sub["_tr"].apply(len).max())
        if max_len < need:
            continue
        padded = np.full((len(sub), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(sub["_tr"].values):
            padded[i, :len(arr)] = arr
        if ctl_pick == "maxgrowth":
            # 성장 = **시간대별 peak 면적 − 초기 면적** (2.5 규약)
            head = padded[:, tix]
            pick = np.nanmax(head, axis=1) - head[:, 0]
        else:
            pick = padded[:, CTL_PICK_INDEX]
        if np.all(~np.isfinite(pick)):
            continue
        best = int(np.nanargmax(pick))
        ctl = padded[best, tix]
        if np.all(np.isfinite(ctl)):
            ctl_map[sid] = ctl.astype(np.float64)

    rows = []
    for (sid, drug), sub in df[df["antimicrobial"] != "Cont"].groupby(
            ["sample_id", "antimicrobial"], sort=False):
        if sid not in ctl_map:
            continue
        ctl = ctl_map[sid]
        for rank, srow in sub.sort_values("concentration").reset_index(drop=True).iterrows():
            traj = np.asarray(srow["_tr"], dtype=np.float64)
            if len(traj) < need or not np.all(np.isfinite(traj[tix])):
                continue
            feat = features_for_row(sid, drug, rank, traj[tix], ctl)
            if label_col:
                feat[label_col] = srow[label_col]
            for c in extra_cols:
                feat[c] = srow[c]
            rows.append(feat)
    return pd.DataFrame(rows)
