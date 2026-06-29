"""genus-level cross-domain ISO transfer — train→FDA (cell/global followup).

train 도메인에서 (genus, antimicrobial) 단위 isotonic 을 fit → FDA 에 genus 로 매칭 적용 후
PASS@0.65 측정. summary 는 organism_group (verify 고정). 기존 cell(74)/global(80)/FDA-cell(103)/
FDA-genus(102) 와 비교해 genus 풀링이 train 전이를 구제하는지 확인.

산출: agent_system/output/cross_domain_iso_genus/dtw_per_conc_routed_train_genus_iso.csv + verify PASS.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from claudeCode.isotonic_lookup import PerCellIsotonicLookup
from claudeCode.organism_normalize import normalize_organism_group

ROOT = Path("/home/kptae/project/qnt_algorithm")
TRAIN_PC = ROOT / "claudeCode/output_dtw_aggregate_train_domain/dtw_per_conc.csv"
FDA_PC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
TRAIN_MAP = "/data/dRAST30_prepare_csv/2024_0709_traintest_allInfo.csv"
VERIFY = ROOT / "claudeCode/verify_method_sir_pipeline.py"
OUTD = ROOT / "agent_system/output/cross_domain_iso_genus"
OG, GEN, AMR, MODEL, LAB = "organism_group", "genus", "antimicrobial", "dtw_model_pred", "gt_gng"


def _fit_by_genus(df: pd.DataFrame) -> PerCellIsotonicLookup:
    """genus 단위 ISO: organism_group 자리에 genus 를 넣어 PerCellIsotonicLookup.fit."""
    w = df.copy()
    w[OG] = w[GEN].astype(str)
    return PerCellIsotonicLookup.fit(w, min_rows=30, min_per_class=3)


def main() -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    # 1) train: genus 복원(train_map) 후 per-genus ISO fit
    train = pd.read_csv(TRAIN_PC, low_memory=False)
    mp = pd.read_csv(TRAIN_MAP, usecols=lambda c: c in ("sample_id", GEN),
                     low_memory=False).drop_duplicates("sample_id")
    mp["sample_id"] = mp["sample_id"].astype(str)
    train["sample_id"] = train["sample_id"].astype(str)
    if GEN in train.columns:
        train = train.drop(columns=[GEN])
    train = train.merge(mp, on="sample_id", how="left")
    n_res = int(train[GEN].notna().sum())
    print(f"[train] rows={len(train)} genus resolved={n_res}")
    tg = _fit_by_genus(train.dropna(subset=[GEN]))
    print(f"[train] per-genus ISO cells = {len(tg.cells)}  (genus×drug)")

    # 2) FDA: genus 로 매칭해 train-genus ISO 적용 (summary 위해 organism_group 보존)
    fda = pd.read_csv(FDA_PC, low_memory=False)
    fda[OG] = fda[OG].map(normalize_organism_group)
    fw = fda.copy()
    fw[OG] = fw[GEN].astype(str)                       # 그룹 키 = genus
    fw = tg.calibrate_dataframe(fw, out_col="_cal")    # train-genus 곡선으로 보정
    out = fda.copy()
    out[MODEL] = fw["_cal"].to_numpy()                 # dtw_model_pred = train-genus 보정값
    if "objarea_crossdomain_pred" in out.columns:
        out = out.drop(columns=["objarea_crossdomain_pred"])
    cal_csv = OUTD / "dtw_per_conc_routed_train_genus_iso.csv"
    out.to_csv(cal_csv, index=False)
    print(f"[saved] {cal_csv}")

    # 3) verify PASS@0.65
    subprocess.run([sys.executable, str(VERIFY), "--per_conc_csv", str(cal_csv),
                    "--output_dir", str(OUTD), "--native_threshold", "0.65"],
                   check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    sm = pd.read_csv(OUTD / "method_summary_model_pred.csv")
    npass = int((sm["FDA_fail_list"] == "PASS").sum())
    nvme = int(sm["FDA_fail_list"].astype(str).str.contains("VME").sum())
    print(f"\nRESULT train_genus_iso PASS@0.65 = {npass}/313  (VME-cell {nvme})")
    print("비교: raw 72 / train-cell 74 / train-global 80 / "
          f"**train-genus {npass}** / FDA-genus 102 / FDA-cell 103")


if __name__ == "__main__":
    main()
