"""Recipe — 방법론 조립 + 운영 SIR/PASS 평가.

Recipe = SCORE(점수 선택/변환) → CALIBRATE(score→P(NG)) → THRESHOLD(이진화) →
운영 verify 파이프라인(EA fix 내장)으로 MIC→SIR→EA/CA/ME/VME → FDA PASS 집계.

예) 운영 권고:
    from agent_system.methods.recipe import Recipe
    from agent_system.methods.score.model_routing import PerCellModelRouting
    from agent_system.methods.calibrate.isotonic import PerCellIsotonic

    rec = Recipe(score=PerCellModelRouting(), calibrate=PerCellIsotonic(), threshold=0.65)
    print(rec.evaluate())          # {'pass': 103, 'vme_cells': .., 'vme_rate': .., ...}

per_conc 를 넘기지 않으면 통합 입력(base.UNIFIED_INPUT)을 자동 로드한다.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method

# verify 의 build_scores 가 참조하는 feature 컬럼(dtw_object_area, dtw_combo_dino_ts_obj 등)을
# 모두 보장하기 위해, 활성 점수만 교체한 전체 base per_conc 를 기록한다 (검증된 harness 방식).
_BASE_PER_CONC = B.ROOT / "claudeCode/output_dtw_aggregate_full/dtw_per_conc.csv"
_BASE_CACHE: Optional[pd.DataFrame] = None


def _base_per_conc() -> pd.DataFrame:
    global _BASE_CACHE
    if _BASE_CACHE is None:
        _BASE_CACHE = pd.read_csv(_BASE_PER_CONC, low_memory=False)
    return _BASE_CACHE.copy()


def load_unified(clean: bool = False) -> pd.DataFrame:
    """통합 입력 로드. clean=True 면 TE(technical error) 제거 평가셋(1434 sample)."""
    path = B.UNIFIED_CLEAN if clean else B.UNIFIED_INPUT
    if not path.exists():
        hint = ("python -m agent_system.methods.build_te_split" if clean
                else "python -m agent_system.methods.build_unified_input")
        raise FileNotFoundError(f"{path} 없음. 먼저 빌드: {hint}")
    return pd.read_parquet(path)


def run_verify(per_conc: pd.DataFrame, out_dir: Path, threshold: float) -> dict:
    """SCORE_COL 이 채워진 per_conc → 운영 verify 파이프라인 → PASS/VME 집계.

    verify 는 내부에서 objarea GBM 을 다시 merge 하므로 충돌 방지를 위해 최소 컬럼만 기록.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # 경계 번역: 활성 점수(ng_score)를 전체 base per_conc 에 key 로 병합해 외부 컬럼
    # dtw_model_pred 를 교체. → build_scores 가 참조하는 모든 feature 컬럼이 보존됨
    # (검증된 harness 와 동일 구조). objarea_crossdomain_pred 는 verify 가 gbm_oof 에서
    # 다시 merge 하므로 base 에 두지 않는다.
    score = per_conc[B.KEY + [B.SCORE_COL]].rename(columns={B.SCORE_COL: "_new_score"})
    full = _base_per_conc()
    # 전달된 per_conc 의 sample 만 평가한다(그 외 base sample 제외). clean 통합입력을 넘기면
    # TE(technical error) sample 이 자동 제외되고, 전체 입력을 넘기면 전부 유지(무변).
    keep_ids = set(per_conc["sample_id"].astype(str))
    full = full[full["sample_id"].astype(str).isin(keep_ids)]
    full = full.merge(score, on=B.KEY, how="left")
    full[B.EXTERNAL_SCORE_COL] = full["_new_score"].where(
        full["_new_score"].notna(), full[B.EXTERNAL_SCORE_COL])
    full = full.drop(columns=["_new_score"])
    if "objarea_crossdomain_pred" in full.columns:
        full = full.drop(columns=["objarea_crossdomain_pred"])
    csv = out_dir / "per_conc.csv"
    full.to_csv(csv, index=False)
    subprocess.run([B.PYTHON, str(B.VERIFY_PIPELINE),
                    "--per_conc_csv", str(csv), "--output_dir", str(out_dir),
                    "--native_threshold", str(threshold)],
                   check=True, cwd=str(B.ROOT), stdout=subprocess.DEVNULL)
    summ = pd.read_csv(out_dir / "method_summary_model_pred.csv")
    feas = pd.read_csv(out_dir / "method_sir_feasibility.csv")
    n_pass = int((summ["FDA_fail_list"] == "PASS").sum())
    n_vme = int(summ["FDA_fail_list"].astype(str).str.contains("VME").sum())
    vme_rate = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    return {"pass": n_pass, "vme_cells": n_vme, "vme_rate": vme_rate,
            "n_cells": int(len(summ)), "summary_csv": str(out_dir / "method_summary_model_pred.csv")}


@dataclass
class Recipe:
    """SCORE + CALIBRATE + THRESHOLD 조립. evaluate() 로 FDA PASS 산출."""
    score: Optional[Method] = None        # Stage.SCORE
    calibrate: Optional[Method] = None    # Stage.CALIBRATE
    threshold: float = 0.65               # Stage.THRESHOLD (native_threshold)
    name: str = "recipe"
    out_subdir: Optional[str] = None      # output/<out_subdir>; 기본 name
    _validated: bool = field(default=False, repr=False)

    def _validate(self) -> None:
        if self.score is not None and self.score.stage is not B.Stage.SCORE:
            raise ValueError(f"score 는 Stage.SCORE 여야 함: {self.score}")
        if self.calibrate is not None and self.calibrate.stage is not B.Stage.CALIBRATE:
            raise ValueError(f"calibrate 는 Stage.CALIBRATE 여야 함: {self.calibrate}")
        self._validated = True

    def apply_chain(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        """SCORE → CALIBRATE 적용 (verify 직전 per_conc 반환). threshold 는 verify 단계."""
        if not self._validated:
            self._validate()
        pc = per_conc.copy()
        if self.score is not None:
            pc = self.score.apply(pc)
        if self.calibrate is not None:
            pc = self.calibrate.apply(pc)
        return pc

    def evaluate(self, per_conc: Optional[pd.DataFrame] = None,
                 out_dir: Optional[Path] = None) -> dict:
        if per_conc is None:
            per_conc = load_unified()
        pc = self.apply_chain(per_conc)
        out = out_dir or (B.OUTPUT_DIR / (self.out_subdir or self.name))
        res = run_verify(pc, Path(out), self.threshold)
        res.update({"recipe": self.name, "threshold": self.threshold,
                    "score": self.score.name if self.score else None,
                    "calibrate": self.calibrate.name if self.calibrate else None})
        return res
