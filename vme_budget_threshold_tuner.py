"""VME budget threshold autotuner (P6, 2026-05-28).

사용자가 VME budget (e.g., ≤ 1.5% FDA 엄격 / ≤ 2.0% 균형 / ≤ 3.5% 적극)을 입력하면
그 budget 만족하면서 PASS를 최대화하는 threshold를 자동 추천.

전략:
  1. 주어진 per_conc CSV (routed + ISO already applied)에 대해 threshold grid sweep
  2. 각 threshold에서 verify_method_sir_pipeline 호출 → (PASS, EA, CA, ME, VME) 측정
  3. 결과 표 + VME budget 필터링 → max PASS recipe 추천
  4. PASS-VME Pareto frontier 시각화 (옵션)

CLI:
    python -m agent_system.vme_budget_threshold_tuner \\
        --per_conc claudeCode/output_subset_eval/dtw_per_conc_routed_iso.csv \\
        --grid 0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80 \\
        --vme_budget 0.020 \\
        --output_dir agent_system/output/threshold_tuner/2026-05-28

산출:
  - sweep_results.csv (threshold × metrics)
  - recommendation.json (budget 만족 best threshold)
  - pareto_frontier.png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/kptae/project/qnt_algorithm")
DEFAULT_GBM = ROOT / "claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv"


def measure_at_threshold(per_conc: Path, threshold: float,
                         output_dir: Path, gbm_oof: Path) -> dict:
    out = output_dir / f"thr_{int(threshold*100):03d}"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "PYTHONPATH": ".",
           "LD_LIBRARY_PATH": "/home/kptae/miniconda3/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
    cmd = ["python", "-m", "claudeCode.verify_method_sir_pipeline",
           "--per_conc_csv", str(per_conc),
           "--gbm_oof_csv", str(gbm_oof),
           "--output_dir", str(out),
           "--native_threshold", str(threshold)]
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        return {"threshold": threshold, "error": r.stderr[-500:]}
    df = pd.read_csv(out / "method_summary_model_pred.csv")
    return {
        "threshold": threshold,
        "fda_pass": int((df["FDA_fail_list_exception_rule"].astype(str).str.strip() == "PASS").sum()),
        "n_cells": len(df),
        "ea_global": float(df["EA_True_Count"].sum() / df["EA_Total"].sum()),
        "ca_global": float(df["CA_True_Count"].sum() / df["CA_Total"].sum()),
        "me_global": float(df["ME_True_Count"].sum() / df["ME_Total"].sum()),
        "vme_global": float(df["VME_True_Count"].sum() / df["VME_Total"].sum()),
        "output_dir": str(out),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per_conc", required=True)
    p.add_argument("--grid", default="0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--vme_budget", type=float, default=0.020,
                   help="VME 상한 (default 0.020 = 2.0%, FDA 엄격 0.015 / 권고 0.020)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gbm_oof", default=str(DEFAULT_GBM))
    args = p.parse_args()

    grid = [float(t) for t in args.grid.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] {len(grid)} thresholds on {args.per_conc}")
    rows = []
    for t in grid:
        print(f"  -> threshold {t}")
        rows.append(measure_at_threshold(Path(args.per_conc), t, out_dir, Path(args.gbm_oof)))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "sweep_results.csv", index=False)
    print(f"\n[save] {out_dir/'sweep_results.csv'}")

    # filter by VME budget
    valid = sweep[sweep["vme_global"] <= args.vme_budget].copy()
    valid = valid.sort_values("fda_pass", ascending=False)
    if len(valid):
        best = valid.iloc[0].to_dict()
        rec = {
            "vme_budget": args.vme_budget,
            "best_threshold": best["threshold"],
            "fda_pass": best["fda_pass"],
            "vme_global": best["vme_global"],
            "ea_global": best["ea_global"],
            "ca_global": best["ca_global"],
            "me_global": best["me_global"],
            "verdict": "FOUND",
        }
    else:
        # no threshold meets budget — recommend min VME
        sweep_sorted = sweep.sort_values("vme_global")
        min_vme = sweep_sorted.iloc[0].to_dict()
        rec = {
            "vme_budget": args.vme_budget,
            "verdict": "NO_THRESHOLD_MEETS_BUDGET",
            "min_vme_threshold": min_vme["threshold"],
            "min_vme": min_vme["vme_global"],
            "min_vme_pass": min_vme["fda_pass"],
            "note": f"No threshold reaches VME ≤ {args.vme_budget*100:.2f}%. "
                    f"min VME found: {min_vme['vme_global']*100:.2f}% at thr={min_vme['threshold']}",
        }
    (out_dir / "recommendation.json").write_text(json.dumps(rec, indent=2),
                                                  encoding="utf-8")
    print(f"\n[recommendation]")
    for k, v in rec.items():
        print(f"  {k:25s} {v}")

    # Pareto plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["vme_global"] * 100, sweep["fda_pass"], "o-",
            color="#1565c0", lw=2, markersize=8)
    for _, r in sweep.iterrows():
        ax.annotate(f"t={r['threshold']}", (r["vme_global"]*100, r["fda_pass"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8.5)
    ax.axvline(args.vme_budget * 100, color="red", ls="--", lw=1,
               label=f"VME budget {args.vme_budget*100:.2f}%")
    ax.set_xlabel("VME global (%)")
    ax.set_ylabel("FDA PASS (/313)")
    ax.set_title(f"Threshold sweep — Pareto frontier (per_conc={Path(args.per_conc).stem})",
                 fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_frontier.png", dpi=130)
    print(f"[save] {out_dir/'pareto_frontier.png'}")


if __name__ == "__main__":
    main()
