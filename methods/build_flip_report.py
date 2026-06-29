"""개선(FAIL→PASS) cell 1건을 단일 HTML 로 — iso 곡선 + 히스토그램 + 개선 샘플 내역.

구성:
  1) isotonic 교정 함수 (organism vs genus 곡선)         ← plot_iso_curve
  2) 점수 분포 히스토그램 (raw vs OOF-iso, verdict 배지)  ← plot_flip_demo
  3) 개선된 샘플 표 (raw→OOF-iso 에서 VME/EA/ME 가 호전된 isolate 별 MIC/SIR)

전제: find_flip_cell.py 가 output/flip_raw, flip_org_oof 를 생성해 둠.
사용: python -m agent_system.methods.build_flip_report --organism "Enterococcus faecium" --drug LNZ
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys

import pandas as pd

from agent_system.methods import base as B

OUT = B.PKG_DIR / "output"
GENUS_SUMMARY = OUT / "genus_insample" / "method_summary_model_pred.csv"
EVAL_COLS = ["sample_id", "bmd_mic", "bmd_sir", "drast_mic", "drast_sir",
             "isEA", "isCA", "isVME", "isME"]


def _b64(path) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()


def _run(mod: str, extra: list[str]) -> None:
    env = dict(os.environ, PYTHONPATH=str(B.ROOT))
    subprocess.run([sys.executable, "-m", mod] + extra, check=True,
                   cwd=str(B.ROOT), env=env, stdout=subprocess.DEVNULL)


def _cell(df, org, drug):
    return df[(df[B.OG_COL] == org) & (df[B.DRUG_COL] == drug)].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", default="Enterococcus faecium")
    ap.add_argument("--drug", default="LNZ")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    org, drug = args.organism, args.drug
    slug = (org.split()[0] + "_" + drug)
    out_html = args.out or str(OUT / f"flip_report_{slug}.html")

    # --- 1) 이미지 생성 (cell 전용) ---
    iso_png = OUT / f"rep_iso_{slug}.png"
    hist_png = OUT / f"rep_hist_{slug}.png"
    _run("agent_system.methods.plot_iso_curve",
         ["--organism", org, "--drug", drug, "--summary_csv", str(GENUS_SUMMARY),
          "--out", str(iso_png)])
    _run("agent_system.methods.plot_flip_demo",
         ["--organism", org, "--drug", drug, "--out", str(hist_png)])

    # --- 2) verdict / 카운트 ---
    def verdict_counts(d):
        sm = _cell(pd.read_csv(OUT / d / "method_summary_model_pred.csv"), org, drug)
        ev = _cell(pd.read_csv(OUT / d / "method_eval_model_pred.csv"), org, drug)
        v = str(sm["FDA_fail_list"].iloc[0])
        c = {k: int(ev[k].sum()) for k in ("isEA", "isCA", "isVME", "isME")}
        return v, c, ev
    v_raw, c_raw, ev_raw = verdict_counts("flip_raw")
    v_iso, c_iso, ev_iso = verdict_counts("flip_org_oof")
    n = len(ev_raw)

    # --- 3) 개선 샘플 (raw→iso 에서 VME/ME 해소 또는 EA 획득) ---
    a = ev_raw[EVAL_COLS].add_suffix("_raw").rename(columns={"sample_id_raw": "sample_id"})
    b = ev_iso[EVAL_COLS].add_suffix("_iso").rename(columns={"sample_id_iso": "sample_id"})
    m = a.merge(b, on="sample_id", how="inner")
    improved = m[((m["isVME_raw"]) & (~m["isVME_iso"]))
                 | ((m["isME_raw"]) & (~m["isME_iso"]))
                 | ((~m["isEA_raw"]) & (m["isEA_iso"]))].copy()

    def _reason(r):
        rs = []
        if r["isVME_raw"] and not r["isVME_iso"]:
            rs.append("VME 해소")
        if r["isME_raw"] and not r["isME_iso"]:
            rs.append("ME 해소")
        if not r["isEA_raw"] and r["isEA_iso"]:
            rs.append("EA 획득")
        return ", ".join(rs)
    improved["reason"] = improved.apply(_reason, axis=1)

    rows = ""
    for _, r in improved.head(60).iterrows():
        rows += (
            f"<tr><td>{r['sample_id']}</td>"
            f"<td>{r['bmd_mic_raw']} / {r['bmd_sir_raw']}</td>"
            f"<td>{r['drast_mic_raw']} / <b>{r['drast_sir_raw']}</b></td>"
            f"<td>{r['drast_mic_iso']} / <b>{r['drast_sir_iso']}</b></td>"
            f"<td class='reason'>{r['reason']}</td></tr>\n")

    def chip(label, raw, iso, good_low=True):
        better = (iso < raw) if good_low else (iso > raw)
        color = "#1b5e20" if better else ("#b71c1c" if (iso > raw) == good_low else "#555")
        return (f"<div class='chip'><span class='cl'>{label}</span>"
                f"<span style='color:{color}'>{raw} → {iso}</span></div>")

    pass_raw = v_raw == "PASS"
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>개선 cell 리포트 — {org} × {drug}</title>
<style>
 body{{font-family:-apple-system,system-ui,'Malgun Gothic',sans-serif;background:#f4f6f9;
   margin:0;color:#222;line-height:1.6}}
 .wrap{{max-width:1000px;margin:0 auto;padding:26px 30px 80px}}
 h1{{font-size:22px;border-bottom:3px solid #2979ff;padding-bottom:9px}}
 h2{{font-size:17px;color:#1565c0;margin:30px 0 8px;border-left:5px solid #2979ff;padding-left:11px}}
 .sub{{color:#666;font-size:13px;margin:6px 0 16px}}
 .verdict{{display:flex;gap:14px;align-items:center;margin:14px 0}}
 .badge{{font-weight:700;border-radius:8px;padding:8px 16px;font-size:15px}}
 .fail{{background:#ffebee;color:#b71c1c;border:1px solid #b71c1c}}
 .pass{{background:#e8f5e9;color:#1b5e20;border:1px solid #1b5e20}}
 .arrow{{font-size:20px;color:#888}}
 .chips{{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 4px}}
 .chip{{background:#fff;border:1px solid #ddd;border-radius:7px;padding:6px 12px;font-size:13px}}
 .chip .cl{{color:#888;margin-right:8px}}
 img{{max-width:100%;border:1px solid #e0e4ea;border-radius:8px;background:#fff}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff;border-radius:8px;
   overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06);margin:8px 0}}
 th,td{{border:1px solid #e3e7ee;padding:6px 9px;text-align:center}}
 th{{background:#eef4ff;color:#0d47a1}}
 td.reason{{color:#1b5e20;font-weight:600;text-align:left}}
 .note{{font-size:12.5px;color:#555;background:#f7f9fc;border:1px solid #e3e7ee;
   border-radius:7px;padding:10px 14px;margin:8px 0}}
 code{{background:#eceff3;padding:1px 6px;border-radius:4px}}
</style></head><body><div class="wrap">
<h1>개선(FAIL→PASS) cell 리포트 — {org} × {drug}</h1>
<p class="sub">routed model_pred → per-cell isotonic(<b>OOF, held-out</b>) → threshold 0.65 ·
샘플 {n}개 · 학습/평가 분리(GroupKFold sample_id)</p>

<div class="verdict">
  <span class="badge fail">raw: {'PASS' if pass_raw else 'FAIL ('+v_raw+')'}</span>
  <span class="arrow">→</span>
  <span class="badge pass">organism-iso OOF: {v_iso}</span>
</div>
<div class="chips">
  {chip("EA pass", c_raw['isEA'], c_iso['isEA'], good_low=False)}
  {chip("VME", c_raw['isVME'], c_iso['isVME'])}
  {chip("ME", c_raw['isME'], c_iso['isME'])}
  {chip("CA", c_raw['isCA'], c_iso['isCA'], good_low=False)}
</div>
<p class="note">cell 판정은 행 오분류 총량이 아니라 <b>샘플별 MIC→SIR 일치율</b>(EA·VME·ME 의 FDA
비율 기준)로 정해진다. 아래 표의 소수 isolate 가 합격선을 가른다.</p>

<h2>1. Isotonic 교정 함수</h2>
<p class="note">x = raw ng_score, y = 보정 P(NG). 단조 계단 함수 → ranking 보존, 확률만 재배치.
organism(파랑) vs genus(주황) 곡선 비교 + 0.65 경계.</p>
<img src="data:image/png;base64,{_b64(iso_png)}">

<h2>2. 점수 분포 히스토그램 (raw vs OOF-iso)</h2>
<p class="note">G(회색)·NG(빨강) 분포가 0.65 경계 양옆으로 어떻게 재배치되는지. 위=raw, 아래=OOF-iso.</p>
<img src="data:image/png;base64,{_b64(hist_png)}">

<h2>3. 개선된 샘플 ({len(improved)} isolate)</h2>
<p class="note">raw→OOF-iso 에서 VME/ME 가 해소되거나 EA 를 획득한 isolate. SIR 은 굵게(예측). 이
샘플들의 MIC 가 reference ±1 dilution 안으로 들어오거나 breakpoint 올바른 쪽으로 이동해 verdict 가 뒤집힘.</p>
<table>
<thead><tr><th>sample_id</th><th>BMD MIC / SIR (기준)</th>
<th>raw 예측 MIC / SIR</th><th>OOF-iso 예측 MIC / SIR</th><th>개선</th></tr></thead>
<tbody>{rows}</tbody></table>

<p style="color:#888;font-size:11.5px;margin-top:26px">
생성: <code>python -m agent_system.methods.build_flip_report --organism "{org}" --drug {drug}</code></p>
</div></body></html>"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"saved -> {out_html}")
    print(f"  raw {v_raw} → iso {v_iso} | improved isolates: {len(improved)}")


if __name__ == "__main__":
    main()
