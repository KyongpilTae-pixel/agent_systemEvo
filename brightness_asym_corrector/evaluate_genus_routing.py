#!/usr/bin/env python3
"""evo 워크북의 genus 단위 모델 맵핑으로 라우팅하고 organism×antimicrobial 단위로 성능 평가.

evo_selected_model_and_brightness_thershold_260529.xlsx 의 (Genus, antimicrobial) → Model
을 각 organism cell 에 부여(genus 의 모델을 그 genus 소속 organism 전부에 적용).
이후 운영 파이프라인 동일: per-cell isotonic → threshold 0.65 → drast_gng→MIC→SIR→evalEA.

비교:
  our_routing   : 우리 AUROC per-organism routing (= routed_iso_t65 baseline, PASS 103)
  genus_routing : evo genus 모델 맵핑 (evo 미지정 cell 은 our_routing fallback)

출력: agent_system/brightness_asym_corrector/artifacts/genus_routing/
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                          # noqa: E402
sys.path.insert(0, str(C.ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

EVO = C.ROOT / "claudeCode/data/evo_selected_model_and_brightness_thershold_260529.xlsx"
SUBSET = C.ROOT / "claudeCode/output_subset_eval"
OUT = C.PKG_DIR / "artifacts" / "genus_routing"
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]
NAME_NORM = {"deployed_model": "same_mic",
             "object_area_045~08_threshold": "object_area_045~08",
             "object_area_all_threshold": "object_area_all"}


def evo_genus_model() -> dict:
    df = pd.read_excel(EVO, "Model selected", header=0).dropna(axis=1, how="all")
    df = df.dropna(subset=["Genus", "antimicrobial", "Model"])
    df["Genus"] = df["Genus"].str.strip()
    df["antimicrobial"] = df["antimicrobial"].str.strip()
    df["m"] = df["Model"].map(lambda m: NAME_NORM.get(str(m).strip(), str(m).strip()))
    return {(g, a): m for g, a, m in zip(df["Genus"], df["antimicrobial"], df["m"])}


def model_pred(model: str) -> pd.DataFrame:
    return pd.read_csv(SUBSET / model / "model_pred.csv",
                       usecols=["sample_id", "antimicrobial", "concentration_idx_0", "model_pred"]
                       ).rename(columns={"model_pred": f"_mp_{model}"})


def calibrate(per_conc: pd.DataFrame) -> pd.DataFrame:
    iso = PerCellIsotonicLookup.fit(per_conc)
    out = iso.calibrate_dataframe(per_conc, out_col="_cal")
    out["dtw_model_pred"] = out["_cal"]
    return out.drop(columns=["_cal"])


def run_sir(per_conc_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(C.VERIFY_PIPELINE), "--per_conc_csv", str(per_conc_csv),
                    "--output_dir", str(out_dir), "--native_threshold", str(C.NATIVE_THRESHOLD)],
                   check=True, cwd=str(C.ROOT), stdout=subprocess.DEVNULL)
    summ = pd.read_csv(out_dir / "method_summary_model_pred.csv")
    feas = pd.read_csv(out_dir / "method_sir_feasibility.csv")
    vr = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    return summ, vr


def pv(summ, tag):
    s = summ[["organism_group", "antimicrobial", "FDA_fail_list"]].copy()
    s[f"{tag}_pass"] = s["FDA_fail_list"] == "PASS"
    s[f"{tag}_vme"] = s["FDA_fail_list"].astype(str).str.contains("VME")
    return s.rename(columns={"FDA_fail_list": f"{tag}_fail"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)
    base["organism_group"] = base["organism_group"].astype(str).str.strip()
    base["antimicrobial"] = base["antimicrobial"].astype(str).str.strip()
    base["genus"] = base["genus"].astype(str).str.strip()

    gmap = evo_genus_model()
    # 각 cell 에 부여할 evo 모델 (genus×drug)
    cell = base[["organism_group", "antimicrobial", "genus"]].drop_duplicates()
    cell["evo_model"] = [gmap.get((g, a)) for g, a in zip(cell["genus"], cell["antimicrobial"])]
    n_cov = int(cell["evo_model"].notna().sum())
    print(f"[genus routing] {len(cell)} cell 중 evo 지정 {n_cov}, fallback {len(cell)-n_cov}")
    print("evo 모델 분포:", cell["evo_model"].value_counts(dropna=False).to_dict())

    # 필요한 모델 예측 로드 후 genus 모델로 dtw_model_pred 덮어쓰기
    models = {m: model_pred(m) for m in cell["evo_model"].dropna().unique()}
    gr = base.copy()
    for _, c in cell.dropna(subset=["evo_model"]).iterrows():
        m = c["evo_model"]
        mask = (gr["organism_group"] == c["organism_group"]) & (gr["antimicrobial"] == c["antimicrobial"])
        if not mask.any():
            continue
        sub = gr.loc[mask, KEY].merge(models[m], on=KEY, how="left")
        gr.loc[mask, "dtw_model_pred"] = sub[f"_mp_{m}"].to_numpy()

    # our routing baseline (= routed_iso_t65)
    base_csv = OUT / "per_conc_our_routing.csv"
    calibrate(base.copy()).to_csv(base_csv, index=False)
    our_summ, our_vr = run_sir(base_csv, OUT / "sir_our_routing")

    gr_csv = OUT / "per_conc_genus_routing.csv"
    calibrate(gr).to_csv(gr_csv, index=False)
    gr_summ, gr_vr = run_sir(gr_csv, OUT / "sir_genus_routing")

    m = pv(our_summ, "our").merge(pv(gr_summ, "genus"),
                                  on=["organism_group", "antimicrobial"], how="outer")
    m.to_csv(OUT / "genus_routing_compare.csv", index=False)
    op, ov = int(m["our_pass"].sum()), int(m["our_vme"].sum())
    gp, gv = int(m["genus_pass"].sum()), int(m["genus_vme"].sum())
    gain = m[(~m["our_pass"]) & m["genus_pass"]]
    loss = m[m["our_pass"] & (~m["genus_pass"])]
    print("\n===== organism×antimicrobial 성능 (전체 313 cell) =====")
    print(f"  our_routing   (AUROC)  PASS {op}  VME-cell {ov}  VME-rate {our_vr:.4f}")
    print(f"  genus_routing (evo)    PASS {gp}  VME-cell {gv}  VME-rate {gr_vr:.4f}   "
          f"ΔPASS {gp-op:+d}(gain {len(gain)}/loss {len(loss)})  ΔVME {gv-ov:+d}")
    print("\n  [genus_routing PASS gain]")
    print(gain[["organism_group", "antimicrobial", "our_fail", "genus_fail"]].to_string(index=False))
    print("  [genus_routing PASS loss]")
    print(loss[["organism_group", "antimicrobial", "our_fail", "genus_fail"]].to_string(index=False))
    (OUT / "manifest.json").write_text(json.dumps({
        "our_routing": {"pass": op, "vme_cells": ov, "vme_rate": round(our_vr, 4)},
        "genus_routing": {"pass": gp, "vme_cells": gv, "vme_rate": round(gr_vr, 4)},
        "delta_pass": gp - op, "delta_vme": gv - ov,
        "pass_gain": int(len(gain)), "pass_loss": int(len(loss)),
        "evo_cells_covered": n_cov,
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT/'genus_routing_compare.csv'}")


if __name__ == "__main__":
    main()
