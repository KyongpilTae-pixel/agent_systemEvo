"""genus-iso 가 VME 를 악화시키는 cell 1건 → 단일 HTML (FAIL→PASS 리포트와 동일 3섹션 형식).

build_flip_report 의 거울상: organism-iso(baseline) → genus-iso(variant) 에서 **악화된 isolate**
(R 균주를 S 로 오판=VME 발생, 또는 EA 상실)를 보인다.

구성: 1) iso 곡선(organism vs genus) 2) 점수 히스토그램(raw/org/genus) 3) 악화 샘플 표.
전제: organism_insample, genus_insample (genus_vs_organism_iso.py 산출).
사용: python -m agent_system.methods.build_vme_regression_report --organism "Staphylococcus aureus" --drug LNZ
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


def _b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def _run(mod, extra):
    env = dict(os.environ, PYTHONPATH=str(B.ROOT))
    subprocess.run([sys.executable, "-m", mod] + extra, check=True, cwd=str(B.ROOT),
                   env=env, stdout=subprocess.DEVNULL)


def _cell(d, org, drug):
    df = pd.read_csv(OUT / d / "method_eval_model_pred.csv")
    return df[(df[B.OG_COL] == org) & (df[B.DRUG_COL] == drug)].copy()


def _verdict_counts(d, org, drug):
    sm = pd.read_csv(OUT / d / "method_summary_model_pred.csv")
    s = sm[(sm[B.OG_COL] == org) & (sm[B.DRUG_COL] == drug)]
    ev = _cell(d, org, drug)
    v = str(s["FDA_fail_list"].iloc[0])
    c = {k: int(ev[k].sum()) for k in ("isEA", "isCA", "isVME", "isME")}
    c["nR"] = int((ev["bmd_sir"] == "R").sum())
    return v, c, ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", default="Staphylococcus aureus")
    ap.add_argument("--drug", default="LNZ")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    org, drug = args.organism, args.drug
    slug = org.split()[0] + "_" + drug
    out_html = args.out or str(OUT / f"vme_regression_{slug}.html")

    iso_png = OUT / f"rep_isoR_{slug}.png"
    hist_png = OUT / f"rep_histR_{slug}.png"
    _run("agent_system.methods.plot_iso_curve",
         ["--organism", org, "--drug", drug, "--summary_csv", str(GENUS_SUMMARY),
          "--out", str(iso_png)])
    _run("agent_system.methods.plot_score_histograms",
         ["--organism", org, "--drug", drug, "--out", str(hist_png)])

    v_org, c_org, ev_org = _verdict_counts("organism_insample", org, drug)
    v_gen, c_gen, ev_gen = _verdict_counts("genus_insample", org, drug)
    n = len(ev_org)

    a = ev_org[EVAL_COLS].add_suffix("_org").rename(columns={"sample_id_org": "sid"})
    b = ev_gen[EVAL_COLS].add_suffix("_gen").rename(columns={"sample_id_gen": "sid"})
    m = a.merge(b, on="sid", how="inner")
    worse = m[((~m["isVME_org"]) & (m["isVME_gen"]))
              | ((m["isEA_org"]) & (~m["isEA_gen"]))
              | ((~m["isME_org"]) & (m["isME_gen"]))].copy()

    def _reason_worse(r):
        rs = []
        if not r["isVME_org"] and r["isVME_gen"]:
            rs.append("VME 발생 (R→S)")
        if r["isEA_org"] and not r["isEA_gen"]:
            rs.append("EA 상실")
        if not r["isME_org"] and r["isME_gen"]:
            rs.append("ME 발생")
        return ", ".join(rs)
    worse["reason"] = worse.apply(_reason_worse, axis=1)
    worse = worse.sort_values("isVME_gen", ascending=False)

    # 개선된 샘플 (genus 가 org 대비 호전)
    improved = m[((m["isVME_org"]) & (~m["isVME_gen"]))
                 | ((~m["isEA_org"]) & (m["isEA_gen"]))
                 | ((m["isME_org"]) & (~m["isME_gen"]))].copy()

    def _reason_imp(r):
        rs = []
        if r["isVME_org"] and not r["isVME_gen"]:
            rs.append("VME 해소")
        if not r["isEA_org"] and r["isEA_gen"]:
            rs.append("EA 획득")
        if r["isME_org"] and not r["isME_gen"]:
            rs.append("ME 해소")
        return ", ".join(rs)
    improved["reason"] = improved.apply(_reason_imp, axis=1)

    def _tbl(df, reason_cls):
        out = ""
        for _, r in df.head(60).iterrows():
            vme = (r["isVME_gen"] and not r["isVME_org"])
            cls = ' class="vme"' if vme else ""
            out += (f"<tr{cls}><td>{r['sid']}</td>"
                    f"<td>{r['bmd_mic_org']} / <b>{r['bmd_sir_org']}</b></td>"
                    f"<td>{r['drast_mic_org']} / {r['drast_sir_org']}</td>"
                    f"<td>{r['drast_mic_gen']} / <b>{r['drast_sir_gen']}</b></td>"
                    f"<td class='{reason_cls}'>{r['reason']}</td></tr>\n")
        return out
    rows = _tbl(worse, "reason")
    rows_imp = _tbl(improved, "reason_good")
    # net 요약 (지표별)
    net = {}
    for k in ("isVME", "isME", "isEA"):
        net[k] = int(m[f"{k}_gen"].sum() - m[f"{k}_org"].sum())

    def chip(label, org_v, gen_v, good_low=True):
        worse_ = (gen_v > org_v) if good_low else (gen_v < org_v)
        color = "#b71c1c" if worse_ else ("#1b5e20" if gen_v != org_v else "#555")
        return (f"<div class='chip'><span class='cl'>{label}</span>"
                f"<span style='color:{color}'>{org_v} → {gen_v}</span></div>")

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VME 악화 cell 리포트 — {org} × {drug}</title>
<style>
 body{{font-family:-apple-system,system-ui,'Malgun Gothic',sans-serif;background:#f4f6f9;margin:0;color:#222;line-height:1.6}}
 .wrap{{max-width:1000px;margin:0 auto;padding:26px 30px 80px}}
 h1{{font-size:22px;border-bottom:3px solid #c62828;padding-bottom:9px}}
 h2{{font-size:17px;color:#b71c1c;margin:30px 0 8px;border-left:5px solid #c62828;padding-left:11px}}
 .sub{{color:#666;font-size:13px;margin:6px 0 16px}}
 .verdict{{display:flex;gap:14px;align-items:center;margin:14px 0}}
 .badge{{font-weight:700;border-radius:8px;padding:8px 16px;font-size:14px}}
 .base{{background:#eef4ff;color:#0d47a1;border:1px solid #1565c0}}
 .vmebad{{background:#ffebee;color:#b71c1c;border:1px solid #b71c1c}}
 .arrow{{font-size:20px;color:#888}}
 .chips{{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}}
 .chip{{background:#fff;border:1px solid #ddd;border-radius:7px;padding:6px 12px;font-size:13px}}
 .chip .cl{{color:#888;margin-right:8px}}
 img{{max-width:100%;border:1px solid #e0e4ea;border-radius:8px;background:#fff}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06);margin:8px 0}}
 th,td{{border:1px solid #e3e7ee;padding:6px 9px;text-align:center}}
 th{{background:#fdecea;color:#b71c1c}}
 tr.vme{{background:#ffebee;font-weight:600}}
 td.reason{{color:#b71c1c;font-weight:600;text-align:left}}
 td.reason_good{{color:#1b5e20;font-weight:600;text-align:left}}
 .net{{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 2px;font-size:13px}}
 .net .it{{background:#fff;border:1px solid #ddd;border-radius:7px;padding:6px 13px}}
 .note{{font-size:12.5px;color:#555;background:#f7f9fc;border:1px solid #e3e7ee;border-radius:7px;padding:10px 14px;margin:8px 0}}
 code{{background:#eceff3;padding:1px 6px;border-radius:4px}}
</style></head><body><div class="wrap">
<h1>VME 악화(organism-iso → genus-iso) 리포트 — {org} × {drug}</h1>
<p class="sub">배포 환경 genus 보정의 위험 사례 · in-sample · 샘플 {n}개 (R {c_org['nR']}개) · threshold 0.65 ·
summary = organism_group</p>

<div class="verdict">
  <span class="badge base">organism-iso: {v_org}</span>
  <span class="arrow">→</span>
  <span class="badge vmebad">genus-iso: {v_gen}</span>
</div>
<div class="chips">
  {chip("VME", c_org['isVME'], c_gen['isVME'])}
  {chip("ME", c_org['isME'], c_gen['isME'])}
  {chip("EA pass", c_org['isEA'], c_gen['isEA'], good_low=False)}
  {chip("CA", c_org['isCA'], c_gen['isCA'], good_low=False)}
</div>
<p class="note">genus 가 species 차이를 뭉개 R 균주의 MIC 를 낮게 잡으면 <b>VME(R→S)</b> 가 새로 생긴다.
VME 는 ME 보다 임상 위험이 크다(내성균을 감수성으로 오판 → 부적절 치료). 아래는 <b>악화·개선
양쪽</b>을 모두 본 순효과다.</p>
<div class="net">
  <span class="it">악화 isolate <b style="color:#b71c1c">{len(worse)}</b> · 개선 isolate
    <b style="color:#1b5e20">{len(improved)}</b></span>
  <span class="it">net VME {net['isVME']:+d}</span>
  <span class="it">net ME {net['isME']:+d}</span>
  <span class="it">net EA {net['isEA']:+d}</span>
</div>

<h2>1. Isotonic 교정 함수 (organism vs genus)</h2>
<p class="note">genus(주황) 곡선이 organism(파랑)보다 NG-aggressive → 실효 threshold 가 낮음 →
경계 R 균주 MIC 를 낮춰 S 로.</p>
<img src="data:image/png;base64,{_b64(iso_png)}">

<h2>2. 점수 분포 히스토그램 (raw / organism-iso / genus-iso)</h2>
<img src="data:image/png;base64,{_b64(hist_png)}">

<h2>3. 악화된 샘플 ({len(worse)} isolate)</h2>
<p class="note">organism-iso→genus-iso 에서 VME 발생/EA 상실/ME 발생. SIR 굵게(예측).
빨강 행 = VME 신규(R 균주를 S 로).</p>
<table>
<thead><tr><th>sample_id</th><th>BMD MIC / SIR (기준)</th>
<th>organism-iso 예측</th><th>genus-iso 예측</th><th>악화</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>4. 개선된 샘플 ({len(improved)} isolate)</h2>
<p class="note">반대로 genus-iso 가 organism-iso 대비 VME 해소/EA 획득/ME 해소한 isolate.
이 cell 의 순효과를 악화 표와 함께 본다.</p>
<table>
<thead><tr><th>sample_id</th><th>BMD MIC / SIR (기준)</th>
<th>organism-iso 예측</th><th>genus-iso 예측</th><th>개선</th></tr></thead>
<tbody>{rows_imp}</tbody></table>

<p style="color:#888;font-size:11.5px;margin-top:24px">생성:
<code>python -m agent_system.methods.build_vme_regression_report --organism "{org}" --drug {drug}</code> ·
전체 위험 목록 = <code>genus_vme_risk_report.html</code></p>
</div></body></html>"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"saved -> {out_html}")
    print(f"  organism-iso {v_org} → genus-iso {v_gen} | VME {c_org['isVME']}→{c_gen['isVME']} | 악화 isolate {len(worse)}")


if __name__ == "__main__":
    main()
