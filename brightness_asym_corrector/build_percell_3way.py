#!/usr/bin/env python3
"""개별 셀별 3-way 방법론 변화 + 성능 비교 보고서 (전체 313 cell).

맵핑 3종 (모델 할당만 다름, 평가 단위·파이프라인 동일):
  기존(baseline)     : same_mic (deployed, 라우팅 이전)  [all cells]
  기존맵핑(our)       : per-organism AUROC routing (cell_model_routing_lookup)
  genus맵핑(genus)    : evo (Genus, 약제) → Model

각 맵핑: per-cell isotonic → t0.65 → drast_gng→MIC→SIR→evalEA → FDA_fail_list per cell.
baseline(same_mic) SIR 은 이 스크립트에서 새로 실행, our/genus 는 genus_routing artifacts 재사용.

출력: artifacts/percell_3way/  (CSV) + report_percell_3way.html
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
GENUS_ART = C.PKG_DIR / "artifacts" / "genus_routing"
OUT = C.PKG_DIR / "artifacts" / "percell_3way"
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]
NN = {"deployed_model": "same_mic", "object_area_045~08_threshold": "object_area_045~08",
      "object_area_all_threshold": "object_area_all"}


def calibrate(pc):
    iso = PerCellIsotonicLookup.fit(pc)
    out = iso.calibrate_dataframe(pc, out_col="_cal")
    out["dtw_model_pred"] = out["_cal"]
    return out.drop(columns=["_cal"])


def run_sir(csv, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(C.VERIFY_PIPELINE), "--per_conc_csv", str(csv),
                    "--output_dir", str(out_dir), "--native_threshold", str(C.NATIVE_THRESHOLD)],
                   check=True, cwd=str(C.ROOT), stdout=subprocess.DEVNULL)
    return pd.read_csv(out_dir / "method_summary_model_pred.csv")


def ensure_baseline_summary(base):
    """same_mic baseline full-panel SIR (없으면 실행)."""
    sm_summ = OUT / "sir_baseline_same_mic" / "method_summary_model_pred.csv"
    if sm_summ.exists():
        return pd.read_csv(sm_summ)
    sm = pd.read_csv(SUBSET / "same_mic" / "model_pred.csv",
                     usecols=KEY + ["model_pred"]).rename(columns={"model_pred": "_sm"})
    pc = base.merge(sm, on=KEY, how="left")
    pc["dtw_model_pred"] = pc["_sm"].where(pc["_sm"].notna(), pc["dtw_model_pred"])
    pc = pc.drop(columns=["_sm"])
    csv = OUT / "per_conc_baseline_same_mic.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    calibrate(pc).to_csv(csv, index=False)
    return run_sir(csv, OUT / "sir_baseline_same_mic")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)
    for c in ("organism_group", "antimicrobial", "genus"):
        base[c] = base[c].astype(str).str.strip()

    # 모델 할당 lookup
    rt = pd.read_csv(SUBSET / "cell_model_routing_lookup.csv")
    our_map = {(o.strip(), a.strip()): m for o, a, m in
               zip(rt["organism_group"], rt["antimicrobial"], rt["best_model"])}
    evo = pd.read_excel(EVO, "Model selected", header=0).dropna(axis=1, how="all").dropna(subset=["Genus", "antimicrobial"])
    evo["Genus"] = evo["Genus"].str.strip(); evo["antimicrobial"] = evo["antimicrobial"].str.strip()
    evo["m"] = evo["Model"].map(lambda x: NN.get(str(x).strip(), str(x).strip()))
    evo_map = {(g, a): m for g, a, m in zip(evo["Genus"], evo["antimicrobial"], evo["m"])}

    # 3 summaries
    base_summ = ensure_baseline_summary(base)
    our_summ = pd.read_csv(GENUS_ART / "sir_our_routing" / "method_summary_model_pred.csv")
    genus_summ = pd.read_csv(GENUS_ART / "sir_genus_routing" / "method_summary_model_pred.csv")

    def fail_map(s):
        return {(o, a): f for o, a, f in zip(s["organism_group"], s["antimicrobial"], s["FDA_fail_list"])}
    fb, fo, fg = fail_map(base_summ), fail_map(our_summ), fail_map(genus_summ)

    cells = base[["organism_group", "antimicrobial", "genus"]].drop_duplicates()
    rows = []
    for og, am, gen in zip(cells["organism_group"], cells["antimicrobial"], cells["genus"]):
        k = (og, am)
        rows.append({
            "organism_group": og, "antimicrobial": am, "genus": gen,
            "model_base": "same_mic", "fail_base": fb.get(k, "—"),
            "model_our": our_map.get(k, "same_mic"), "fail_our": fo.get(k, "—"),
            "model_genus": evo_map.get((gen, am), "same_mic"), "fail_genus": fg.get(k, "—"),
        })
    df = pd.DataFrame(rows)
    for tag in ("base", "our", "genus"):
        df[f"pass_{tag}"] = df[f"fail_{tag}"] == "PASS"
    df.to_csv(OUT / "percell_3way_compare.csv", index=False)

    summary = {t: int(df[f"pass_{t}"].sum()) for t in ("base", "our", "genus")}
    n = len(df)
    print("===== 전체", n, "cell 3-way PASS =====")
    for t, lbl in [("base", "기존(same_mic)"), ("our", "기존맵핑(AUROC)"), ("genus", "genus맵핑(evo)")]:
        print(f"  {lbl:20s}: {summary[t]}")

    _write_html(df, summary, n)
    (OUT / "manifest.json").write_text(json.dumps({"n_cells": n, "pass": summary},
                                                  ensure_ascii=False, indent=2))
    print(f"[saved] {OUT/'percell_3way_compare.csv'}")


def _write_html(df, summary, n):
    import html

    def badge(v):
        v = str(v)
        if v == "PASS":
            return '<span class="pass">PASS</span>'
        if v == "—":
            return '<span class="na">—</span>'
        cls = "vme" if "VME" in v else "fail"
        return f'<span class="{cls}">{html.escape(v)}</span>'

    # 정렬: 맵핑간 결과가 다른 cell 우선
    df = df.copy()
    df["_diff"] = ~((df["fail_base"] == df["fail_our"]) & (df["fail_our"] == df["fail_genus"]))
    df = df.sort_values(["_diff", "genus", "organism_group", "antimicrobial"],
                        ascending=[False, True, True, True])
    rows = ""
    for _, r in df.iterrows():
        ch_o = "" if r["model_base"] == r["model_our"] else "→"
        ch_g = "" if r["model_our"] == r["model_genus"] else "→"
        hi = ' style="background:#fff8e1"' if r["_diff"] else ""
        rows += (
            f"<tr{hi}><td>{html.escape(r['genus'])}</td>"
            f"<td>{html.escape(r['organism_group'])}</td><td>{html.escape(r['antimicrobial'])}</td>"
            f"<td>{html.escape(r['model_base'])}</td><td>{badge(r['fail_base'])}</td>"
            f"<td>{ch_o} {html.escape(r['model_our'])}</td><td>{badge(r['fail_our'])}</td>"
            f"<td>{ch_g} {html.escape(r['model_genus'])}</td><td>{badge(r['fail_genus'])}</td></tr>")

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>개별 셀별 3-way 방법론·성능 비교</title>
<style>
 body{{font-family:-apple-system,'Segoe UI','Malgun Gothic',sans-serif;margin:0;padding:26px 34px;color:#1f2328;}}
 h1{{font-size:22px;margin:0 0 4px;}} .sub{{color:#57606a;font-size:12.5px;}}
 .cards{{display:flex;gap:14px;margin:14px 0;}}
 .card{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:10px;padding:12px 22px;text-align:center;font-size:12.5px;color:#57606a;}}
 .card .v{{font-size:25px;font-weight:700;color:#1f2328;}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0;}}
 th,td{{border:1px solid #d8dee4;padding:4px 7px;text-align:left;}} th{{background:#f0f3f6;position:sticky;top:0;}}
 .pass{{background:#dafbe1;color:#1a7f37;padding:1px 5px;border-radius:4px;font-weight:600;}}
 .fail{{background:#fff1e5;color:#9a6700;padding:1px 5px;border-radius:4px;}}
 .vme{{background:#ffebe9;color:#cf222e;padding:1px 5px;border-radius:4px;font-weight:600;}}
 .na{{color:#afb8c1;}} .note{{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;margin:10px 0;font-size:13px;}}
</style></head><body>
<h1>개별 셀별 3-way 방법론 변화 · 성능 비교</h1>
<div class="sub">전체 {n} cell · 모델 맵핑만 다름, 평가 단위(organism×약제)·파이프라인(per-cell ISO→t0.65→SIR) 동일 · 2026-06-02</div>
<div class="cards">
 <div class="card"><div class="v">{summary['base']}</div>기존 (same_mic)</div>
 <div class="card"><div class="v">{summary['our']}</div>기존맵핑 (AUROC routing)</div>
 <div class="card"><div class="v">{summary['genus']}</div>genus맵핑 (evo)</div>
</div>
<div class="note">
<b>기존맵핑(AUROC)이 PASS 최대.</b> genus맵핑(evo)은 보수적(gram_negative 제외)이라 VME 안전하나 PASS 낮음.
<code>→</code> 표시는 직전 맵핑 대비 모델 변경. 노란 행 = 세 맵핑 결과가 갈리는 cell.
<span class="pass">PASS</span> / <span class="fail">FAIL</span> / <span class="vme">VME 포함</span>
</div>
<table>
<tr><th>genus</th><th>organism_group</th><th>약제</th>
<th>기존 모델</th><th>기존 결과</th><th>기존맵핑 모델</th><th>기존맵핑 결과</th>
<th>genus맵핑 모델</th><th>genus맵핑 결과</th></tr>
{rows}
</table></body></html>"""
    out = OUT.parent.parent / "report_percell_3way.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[saved] {out}  ({len(doc)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
