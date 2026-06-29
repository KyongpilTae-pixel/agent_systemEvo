"""Recipe별 dtw_per_conc CSV 빌드 helper.

기존 dtw_per_conc.csv의 dtw_model_pred 컬럼을 recipe에 따라 치환:

  recipe="baseline_same_mic"
      → 그대로 (same_mic.pt 점수)
  recipe="routed_per_cell"
      → cell마다 best subset 모델 점수 (PerCellModelRouter)
  recipe="shifted_rankavg3"  ← Cycle 1 C1
      → shifted cells: rank_avg(beta_lactam + gram_negative + objarea_gbm_full)
      → other cells: same_mic.pt (baseline)

CLI 사용:
  python -m agent_system.build_per_conc_for_recipe \
      --recipe shifted_rankavg3 \
      --output claudeCode/output_subset_eval/dtw_per_conc_C1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PER_CONC_BASE = Path("claudeCode/output_dtw_aggregate_full/dtw_per_conc.csv")
SUBSET_DIR = Path("claudeCode/output_subset_eval")
GBM_OOF = Path("claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv")
SHIFTED_DIFF = Path("claudeCode/output_dataset_diff_traintest_normalized/dataset_cell_diff_summary.csv")
ROUTED_CSV = SUBSET_DIR / "routed_model_pred.csv"


def _load_subset_pred(model: str) -> pd.DataFrame:
    f = SUBSET_DIR / model / "model_pred.csv"
    df = pd.read_csv(f, low_memory=False)
    return df.rename(columns={"model_pred": f"pred_{model}"})[
        ["sample_id", "antimicrobial", "concentration_idx_0", f"pred_{model}"]]


def _shifted_set() -> set[tuple[str, str]]:
    d = pd.read_csv(SHIFTED_DIFF)
    return {(str(r.organism_group), str(r.antimicrobial))
            for _, r in d[d.bucket == "shifted"].iterrows()}


def build(recipe: str, output: Path) -> dict:
    base = pd.read_csv(PER_CONC_BASE, low_memory=False)
    n = len(base)
    info = {"recipe": recipe, "n_rows": n}

    if recipe == "baseline_same_mic":
        # dtw_model_pred 유지 (same_mic.pt)
        out = base
        info["note"] = "dtw_model_pred = same_mic.pt 그대로"

    elif recipe == "routed_per_cell":
        rt = pd.read_csv(ROUTED_CSV, low_memory=False,
                         usecols=["sample_id", "antimicrobial",
                                  "concentration_idx_0", "model_pred"])
        rt = rt.rename(columns={"model_pred": "routed_pred"})
        m = base.merge(rt, on=["sample_id", "antimicrobial", "concentration_idx_0"],
                       how="left")
        m["dtw_model_pred"] = m["routed_pred"].where(
            m["routed_pred"].notna(), m["dtw_model_pred"])
        out = m.drop(columns=["routed_pred"])
        info["matched_routed"] = int(m["routed_pred"].notna().sum())

    elif recipe == "shifted_rankavg3":
        # 3 model preds + GBM
        models = ["beta_lactam", "gram_negative"]
        sm = base[["sample_id", "antimicrobial", "concentration_idx_0"]].copy()
        for mdl in models:
            sm = sm.merge(_load_subset_pred(mdl),
                          on=["sample_id", "antimicrobial", "concentration_idx_0"],
                          how="left")
        gbm = pd.read_csv(GBM_OOF, low_memory=False,
                          usecols=["sample_id", "antimicrobial", "concentration_idx_0",
                                   "objarea_crossdomain_pred"])
        gbm = gbm.rename(columns={"objarea_crossdomain_pred": "pred_objarea_gbm_full"})
        sm = sm.merge(gbm, on=["sample_id", "antimicrobial", "concentration_idx_0"],
                      how="left")
        rank_cols = ["pred_beta_lactam", "pred_gram_negative", "pred_objarea_gbm_full"]
        sm["rankavg3"] = sm[rank_cols].rank(pct=True).mean(axis=1)

        # shifted cell 식별
        shifted = _shifted_set()
        # base에 organism_group + bucket 매핑
        m = base.merge(sm[["sample_id", "antimicrobial", "concentration_idx_0",
                           "rankavg3"]],
                       on=["sample_id", "antimicrobial", "concentration_idx_0"],
                       how="left")
        is_shifted = m.apply(
            lambda r: (str(r["organism_group"]), str(r["antimicrobial"])) in shifted,
            axis=1)
        n_shifted_rows = int(is_shifted.sum())
        n_replaced = int((is_shifted & m["rankavg3"].notna()).sum())
        # shifted cell + rankavg3 가능 → rank-avg, 나머지 → baseline 유지
        m["dtw_model_pred"] = np.where(
            is_shifted & m["rankavg3"].notna(),
            m["rankavg3"],
            m["dtw_model_pred"],
        )
        out = m.drop(columns=["rankavg3"])
        info.update({"n_shifted_rows": n_shifted_rows,
                     "n_rows_replaced": n_replaced})

    else:
        raise ValueError(f"unknown recipe: {recipe}")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    info["output"] = str(output)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True,
                   choices=["baseline_same_mic", "routed_per_cell", "shifted_rankavg3"])
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    info = build(args.recipe, args.output)
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
