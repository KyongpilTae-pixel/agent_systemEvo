#!/usr/bin/env python3
"""Brightness=True cell에서 brightness GBM 앙상블이 image-only 대비 운영 PASS/VME를
개선하는지 검증 (object_area 판 brightness_ensemble_test.py 의 brightness 버전).

aux 신호 = brightness GBM (train CSV in-domain fit → FDA 추론, brightness_crossdomain_oof.csv).
변형: image_only / prob_ens(0.5*img+0.5*brightness) / brightness_only.
각 변형에 per-cell isotonic(FDA fit) → threshold 0.65 → drast_gng→MIC→SIR→evalEA.
출력: agent_system/output/brightness_feature_ensemble/
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kptae/project/qnt_algorithm")
sys.path.insert(0, str(ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

WORKBOOK = ROOT / "claudeCode/data/F21_GN_Selecetd_model_260123 (3) (1).xlsx"
BASE_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
BRIGHT_OOF = ROOT / "agent_system/output/brightness_gbm/brightness_crossdomain_oof.csv"
OUT_DIR = ROOT / "agent_system/output/brightness_feature_ensemble"
VERIFY = ROOT / "claudeCode/verify_method_sir_pipeline.py"
NATIVE_THRESHOLD = 0.65
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]


def brightness_true_cells() -> set:
    wb = pd.read_excel(WORKBOOK, "Model selected", header=0).dropna(axis=1, how="all")
    wb = wb[["organism_group", "antimicrobial", "Brightness"]].dropna(subset=["organism_group"])
    wb["organism_group"] = wb["organism_group"].astype(str).str.strip()
    wb["antimicrobial"] = wb["antimicrobial"].astype(str).str.strip()
    t = wb[wb["Brightness"] == True]  # noqa: E712
    return set(zip(t["organism_group"], t["antimicrobial"]))


def calibrate_variant(base: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    df = base.copy()
    df["dtw_model_pred"] = np.clip(score, 0.0, 1.0)
    iso = PerCellIsotonicLookup.fit(df)
    df = iso.calibrate_dataframe(df, out_col="_cal")
    df["dtw_model_pred"] = df["_cal"]
    return df.drop(columns=["_cal"])


def run_pipeline(per_conc_csv: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(VERIFY), "--per_conc_csv", str(per_conc_csv),
           "--output_dir", str(out_dir), "--native_threshold", str(NATIVE_THRESHOLD)]
    print("[run]", out_dir.name)
    subprocess.run(cmd, check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    return pd.read_csv(out_dir / "method_summary_model_pred.csv")


def pass_vme(summ: pd.DataFrame) -> pd.DataFrame:
    s = summ[["organism_group", "antimicrobial", "FDA_fail_list"]].copy()
    s["pass"] = s["FDA_fail_list"] == "PASS"
    s["vme"] = s["FDA_fail_list"].astype(str).str.contains("VME")
    return s


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = brightness_true_cells()
    base_full = pd.read_csv(BASE_PER_CONC, low_memory=False)
    base = base_full[base_full.apply(
        lambda r: (str(r["organism_group"]).strip(),
                   str(r["antimicrobial"]).strip()) in cells, axis=1)].copy()
    n_cells = base.groupby(["organism_group", "antimicrobial"]).ngroups
    print(f"[base] Brightness=True {n_cells} cells / {len(base)} rows")

    br = pd.read_csv(BRIGHT_OOF, usecols=KEY + ["brightness_pred"])
    base = base.merge(br, on=KEY, how="left")
    miss = int(base["brightness_pred"].isna().sum())
    if miss:
        print(f"[warn] {miss} rows missing brightness_pred → fallback to model_pred")
        base["brightness_pred"] = base["brightness_pred"].fillna(base["dtw_model_pred"])

    img = base["dtw_model_pred"].astype(float).to_numpy()
    brp = base["brightness_pred"].astype(float).to_numpy()
    # weight sweep on brightness contribution w_br (image weight = 1 - w_br)
    variants = {"image_only": img}
    for w in (0.1, 0.2, 0.3, 0.5):
        variants[f"ens_br{int(w*100):02d}"] = (1 - w) * img + w * brp
    variants["brightness_only"] = brp

    summaries = {}
    for name, score in variants.items():
        pc = calibrate_variant(base.drop(columns=["brightness_pred"]), score)
        csv = OUT_DIR / f"per_conc_{name}.csv"
        pc.to_csv(csv, index=False)
        summaries[name] = pass_vme(run_pipeline(csv, OUT_DIR / f"sir_{name}"))

    merged = None
    for name, s in summaries.items():
        cols = s.rename(columns={"pass": f"{name}_pass", "vme": f"{name}_vme",
                                 "FDA_fail_list": f"{name}_fail"})
        merged = cols if merged is None else merged.merge(
            cols, on=["organism_group", "antimicrobial"], how="outer")
    merged.to_csv(OUT_DIR / "brightness_feature_compare.csv", index=False)

    print("\n===== Brightness=True cells: variant PASS / VME =====")
    rows = []
    for name in variants:
        p = int(merged[f"{name}_pass"].sum()); v = int(merged[f"{name}_vme"].sum())
        print(f"  {name:16s} PASS {p:3d}/{len(merged)}   VME cells {v}")
        rows.append({"variant": name, "pass": p, "vme_cells": v, "n": int(len(merged))})

    ens_names = [v for v in variants if v.startswith("ens_br")]
    print("\n===== 각 가중 변형 vs image_only (net PASS / net VME-cell) =====")
    h2h = []
    img_pass = int(merged["image_only_pass"].sum()); img_vme = int(merged["image_only_vme"].sum())
    for name in ens_names:
        gain = merged[(~merged["image_only_pass"]) & merged[f"{name}_pass"]]
        loss = merged[merged["image_only_pass"] & (~merged[f"{name}_pass"])]
        vfix = merged[merged["image_only_vme"] & (~merged[f"{name}_vme"])]
        vnew = merged[(~merged["image_only_vme"]) & merged[f"{name}_vme"]]
        net_pass = int(merged[f"{name}_pass"].sum()) - img_pass
        net_vme = int(merged[f"{name}_vme"].sum()) - img_vme
        print(f"  {name:10s}  ΔPASS {net_pass:+d} (gain {len(gain)}/loss {len(loss)})   "
              f"ΔVME-cell {net_vme:+d} (fix {len(vfix)}/new {len(vnew)})")
        h2h.append({"variant": name, "net_pass": net_pass, "net_vme_cells": net_vme,
                    "pass_gain": int(len(gain)), "pass_loss": int(len(loss)),
                    "vme_fixed": int(len(vfix)), "vme_new": int(len(vnew))})

    best = max(h2h, key=lambda r: (r["net_pass"], -r["net_vme_cells"]))
    print(f"\n  최선 가중: {best['variant']}  ΔPASS {best['net_pass']:+d}  "
          f"ΔVME-cell {best['net_vme_cells']:+d}")

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "n_cells": int(len(merged)), "native_threshold": NATIVE_THRESHOLD,
        "aux_signal": "brightness GBM (train-CSV in-domain fit, FDA infer)",
        "image_only_pass": img_pass, "image_only_vme_cells": img_vme,
        "variants": rows, "head_to_head_vs_image": h2h, "best": best,
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'brightness_feature_compare.csv'}")


if __name__ == "__main__":
    main()
