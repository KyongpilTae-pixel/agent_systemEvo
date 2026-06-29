#!/usr/bin/env python3
"""LNZ cell 3-way 모델 맵핑 비교 — 기존(baseline) / 기존맵핑(our AUROC) / genus맵핑(evo).

같은 평가 단위(organism×LNZ)·같은 파이프라인(per-cell isotonic → t0.65 → SIR/evalEA),
모델 맵핑만 3가지:
  baseline      : same_mic (deployed, 라우팅 이전)
  our_mapping   : per-organism AUROC routing (cell_model_routing_lookup)
  genus_mapping : evo (Genus, LNZ) → Model

출력: agent_system/brightness_asym_corrector/artifacts/lnz_3way/
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                          # noqa: E402
sys.path.insert(0, str(C.ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

EVO = C.ROOT / "claudeCode/data/evo_selected_model_and_brightness_thershold_260529.xlsx"
SUBSET = C.ROOT / "claudeCode/output_subset_eval"
ROUTING = SUBSET / "cell_model_routing_lookup.csv"
OUT = C.PKG_DIR / "artifacts" / "lnz_3way"
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]
NN = {"deployed_model": "same_mic", "object_area_045~08_threshold": "object_area_045~08",
      "object_area_all_threshold": "object_area_all"}
DRUG = "LNZ"


def model_pred(model: str) -> pd.DataFrame:
    return pd.read_csv(SUBSET / model / "model_pred.csv",
                       usecols=KEY + ["model_pred"]).rename(columns={"model_pred": f"_mp_{model}"})


def calibrate(pc: pd.DataFrame) -> pd.DataFrame:
    iso = PerCellIsotonicLookup.fit(pc)
    out = iso.calibrate_dataframe(pc, out_col="_cal")
    out["dtw_model_pred"] = out["_cal"]
    return out.drop(columns=["_cal"])


def run_sir(csv: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(C.VERIFY_PIPELINE), "--per_conc_csv", str(csv),
                    "--output_dir", str(out_dir), "--native_threshold", str(C.NATIVE_THRESHOLD)],
                   check=True, cwd=str(C.ROOT), stdout=subprocess.DEVNULL)
    return pd.read_csv(out_dir / "method_summary_model_pred.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)
    for c in ("organism_group", "antimicrobial", "genus"):
        base[c] = base[c].astype(str).str.strip()
    lnz = base[base["antimicrobial"] == DRUG].copy()
    cells = lnz[["organism_group", "genus"]].drop_duplicates()
    print(f"[LNZ] FDA 패널 LNZ cell {len(cells)}개: {sorted(cells['organism_group'])}")

    # 맵핑별 모델 할당
    rt = pd.read_csv(ROUTING)
    rt = rt[rt["antimicrobial"] == DRUG]
    our_map = {og: m for og, m in zip(rt["organism_group"].str.strip(), rt["best_model"])}
    evo = pd.read_excel(EVO, "Model selected", header=0).dropna(axis=1, how="all").dropna(subset=["Genus", "antimicrobial"])
    evo["Genus"] = evo["Genus"].str.strip(); evo["antimicrobial"] = evo["antimicrobial"].str.strip()
    evo["m"] = evo["Model"].map(lambda x: NN.get(str(x).strip(), str(x).strip()))
    evo_map = {g: m for g, a, m in zip(evo["Genus"], evo["antimicrobial"], evo["m"]) if a == DRUG}

    assign = {}
    for og, gen in zip(cells["organism_group"], cells["genus"]):
        assign[og] = {"baseline": "same_mic",
                      "our_mapping": our_map.get(og, "same_mic"),
                      "genus_mapping": evo_map.get(gen, "same_mic")}
    print("[assign]")
    for og, a in assign.items():
        print(f"  {og:36s} base={a['baseline']:10s} our={a['our_mapping']:18s} genus={a['genus_mapping']}")

    needed = {m for a in assign.values() for m in a.values()}
    models = {m: model_pred(m) for m in needed}

    summ = {}
    for mapping in ("baseline", "our_mapping", "genus_mapping"):
        pc = lnz.copy()
        for og, a in assign.items():
            m = a[mapping]
            mask = pc["organism_group"] == og
            sub = pc.loc[mask, KEY].merge(models[m], on=KEY, how="left")
            pc.loc[mask, "dtw_model_pred"] = sub[f"_mp_{m}"].to_numpy()
        csv = OUT / f"per_conc_{mapping}.csv"
        calibrate(pc).to_csv(csv, index=False)
        s = run_sir(csv, OUT / f"sir_{mapping}")
        summ[mapping] = s.set_index("organism_group")["FDA_fail_list"].to_dict()

    # 3-way 표
    rows = []
    for og in cells["organism_group"]:
        rows.append({
            "organism_group": og,
            "model_baseline": assign[og]["baseline"],
            "fail_baseline": summ["baseline"].get(og, "—"),
            "model_our": assign[og]["our_mapping"],
            "fail_our": summ["our_mapping"].get(og, "—"),
            "model_genus": assign[og]["genus_mapping"],
            "fail_genus": summ["genus_mapping"].get(og, "—"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lnz_3way_compare.csv", index=False)

    def npass(key):
        return int(sum(v == "PASS" for v in summ[key].values()))
    print("\n===== LNZ 5 cell 3-way PASS =====")
    print(f"  기존(baseline same_mic) : {npass('baseline')}/{len(cells)}")
    print(f"  기존맵핑(our AUROC)      : {npass('our_mapping')}/{len(cells)}")
    print(f"  genus맵핑(evo)           : {npass('genus_mapping')}/{len(cells)}")
    print()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    (OUT / "manifest.json").write_text(json.dumps({
        "n_lnz_cells": int(len(cells)),
        "pass": {k: npass(k) for k in ("baseline", "our_mapping", "genus_mapping")},
        "rows": rows,
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT/'lnz_3way_compare.csv'}")


if __name__ == "__main__":
    main()
