"""genus-iso VME 안전성 위험 스캔 → 단일 HTML 보고서.

genus_vme_scan 의 per-cell 비교를 표/요약 + S.aureus×LNZ iso 곡선(예시)로 묶는다.
산출: output/genus_vme_risk_report.html
사용:  python -m agent_system.methods.build_genus_vme_report
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.genus_vme_scan import _percell, RUNS, KEY

OUT = B.PKG_DIR / "output"


def _b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def _mode_data(od, gd):
    o = _percell(od).add_suffix("_org").rename(columns={f"{k}_org": k for k in KEY})
    g = _percell(gd).add_suffix("_gen").rename(columns={f"{k}_gen": k for k in KEY})
    m = o.merge(g, on=KEY, how="outer")
    m["dVME"] = m["VME_gen"].fillna(0) - m["VME_org"].fillna(0)
    m["pass2vme"] = (m["FDA_fail_list_org"].astype(str) == "PASS") & \
                    (m["FDA_fail_list_gen"].astype(str).str.contains("VME"))
    return m


def _rows(risk):
    out = ""
    for _, r in risk.iterrows():
        rate = r["VME_gen"] / r["nR_gen"] if r["nR_gen"] else float("nan")
        reg = (" 🔴PASS→VME" if r["pass2vme"] else "")
        cls = ' class="reg"' if r["pass2vme"] else ""
        out += (f"<tr{cls}><td>{r['organism_group']}</td><td>{r['antimicrobial']}</td>"
                f"<td>{int(r['nR_gen'])}</td>"
                f"<td>{int(r['VME_org'] or 0)} → <b>{int(r['VME_gen'] or 0)}</b></td>"
                f"<td>{rate*100:.1f}%</td>"
                f"<td class='v'>{r['FDA_fail_list_org']} → {r['FDA_fail_list_gen']}{reg}</td></tr>\n")
    return out


def main() -> None:
    # S.aureus×LNZ iso 곡선(예시) 생성
    iso_png = OUT / "iso_curve_SaureusLNZ.png"
    if not iso_png.exists():
        env = dict(os.environ, PYTHONPATH=str(B.ROOT))
        subprocess.run([sys.executable, "-m", "agent_system.methods.plot_iso_curve",
                        "--organism", "Staphylococcus aureus", "--drug", "LNZ",
                        "--summary_csv", str(OUT / "genus_insample/method_summary_model_pred.csv"),
                        "--out", str(iso_png)], check=True, cwd=str(B.ROOT), env=env,
                       stdout=subprocess.DEVNULL)

    sections = ""
    agg = {}
    for mode, (od, gd) in RUNS.items():
        m = _mode_data(od, gd)
        tot_o = int(m["VME_org"].fillna(0).sum()); tot_g = int(m["VME_gen"].fillna(0).sum())
        risk = m[m["dVME"] > 0].sort_values("dVME", ascending=False)
        n_new = int(m["FDA_fail_list_gen"].astype(str).str.contains("VME").sum()
                    - m["FDA_fail_list_org"].astype(str).str.contains("VME").sum())
        n_reg = int(risk["pass2vme"].sum())
        agg[mode] = dict(tot_o=tot_o, tot_g=tot_g, n_risk=len(risk), n_reg=n_reg)
        label = "in-sample (배포 자산 기준)" if mode == "insample" else "OOF (held-out · 정직)"
        sections += f"""
<h2>{label}</h2>
<p class="note">총 VME isolate <b>organism {tot_o} → genus {tot_g}</b>
(Δ{tot_g-tot_o:+d}). ΔVME&gt;0 cell <b>{len(risk)}개</b>,
그 중 PASS→VME 회귀 <b>{n_reg}개</b>.</p>
<table><thead><tr><th>organism_group</th><th>약제</th><th>R 수</th>
<th>VME (org→gen)</th><th>genus VME율</th><th>verdict (org → gen)</th></tr></thead>
<tbody>{_rows(risk)}</tbody></table>
"""

    head = (f"in-sample 총 VME +{agg['insample']['tot_g']-agg['insample']['tot_o']} "
            f"(악화) · OOF 총 VME {agg['oof']['tot_g']-agg['oof']['tot_o']:+d} "
            f"(개선) — 단 cell-level 위험 {agg['oof']['n_risk']}(OOF)/{agg['insample']['n_risk']}(in)")

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>genus-iso VME 안전성 위험 스캔</title>
<style>
 body{{font-family:-apple-system,system-ui,'Malgun Gothic',sans-serif;background:#f4f6f9;margin:0;color:#222;line-height:1.6}}
 .wrap{{max-width:1020px;margin:0 auto;padding:26px 30px 80px}}
 h1{{font-size:22px;border-bottom:3px solid #c62828;padding-bottom:9px}}
 h2{{font-size:17px;color:#b71c1c;margin:28px 0 8px;border-left:5px solid #c62828;padding-left:11px}}
 .sub{{color:#666;font-size:13px;margin:6px 0 14px}}
 .warn{{background:#fff3e0;border:1px solid #e65100;border-radius:8px;padding:13px 17px;margin:14px 0;font-size:13.5px}}
 .warn b{{color:#bf360c}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06);margin:8px 0}}
 th,td{{border:1px solid #e3e7ee;padding:6px 9px;text-align:center}}
 th{{background:#fdecea;color:#b71c1c}}
 td.v{{text-align:left;font-size:11.5px}}
 tr.reg{{background:#ffebee;font-weight:600}}
 .note{{font-size:12.5px;color:#555;background:#f7f9fc;border:1px solid #e3e7ee;border-radius:7px;padding:10px 14px;margin:8px 0}}
 img{{max-width:100%;border:1px solid #e0e4ea;border-radius:8px;background:#fff}}
 code{{background:#eceff3;padding:1px 6px;border-radius:4px}}
</style></head><body><div class="wrap">
<h1>genus-iso VME 안전성 위험 스캔</h1>
<p class="sub">organism-iso 대비 genus-iso 가 R 균주를 S 로 오판(VME)하는 cell 비교 ·
in-sample(배포 자산) + OOF(held-out) · summary = organism_group×antimicrobial · threshold 0.65</p>

<div class="warn">
<b>요약</b>: genus-iso 는 aggregate held-out PASS +4 / OOF 총 VME {agg['oof']['tot_g']-agg['oof']['tot_o']:+d}
(개선)지만, <b>특정 cell 에서 VME 를 새로 만든다</b>. in-sample 위험 cell {agg['insample']['n_risk']}개
(PASS→VME 회귀 {agg['insample']['n_reg']}개), OOF 위험 cell {agg['oof']['n_risk']}개.
최악 = <b>S. aureus × LNZ (VME 0→4, 17%)</b>. → 운영은 <b>VME 위험 cell 은 organism-iso 로 남기는
하이브리드</b> 권고.
</div>
{sections}

<h2>대표 사례 — S. aureus × LNZ 의 S-leaning 보정</h2>
<p class="note">genus(Staphylococcus, n=3228) 곡선이 organism(S.aureus, n=864)보다 NG-aggressive →
실효 threshold 가 낮음(~0.47 vs ~0.65) → 경계 R 균주의 MIC 를 낮게 잡아 S 로(VME 4/23=17%).</p>
<img src="data:image/png;base64,{_b64(iso_png)}">

<h2>해석 · 권고</h2>
<div class="note">
<ol>
<li><b>aggregate ≠ cell 안전</b>: genus 는 전체 VME 를 (OOF) 줄이지만, S.aureus(LNZ/OXA/PEN)·
Proteus·E.faecium×AMP·K.pneumoniae 등에서 국소 VME 를 새로 만든다.</li>
<li><b>회귀(PASS→VME)</b>: organism 에선 PASS 였는데 genus 로 VME FAIL 되는 cell 존재 — 직접 손해.</li>
<li><b>권고</b>: genus-iso 전면 채택 대신 <b>하이브리드</b> — 일반 cell 은 genus(일반화 우위),
VME 위험 cell(특히 gram-positive S.aureus·Proteus)은 organism-iso 유지. 위험 목록 =
<code>output/genus_vme_risk.csv</code>.</li>
<li>VME 율이 높아도 R 표본이 1~2 인 Proteus cell 은 통계적 불안정 — 별도 표기.</li>
</ol>
</div>
<p style="color:#888;font-size:11.5px;margin-top:24px">생성:
<code>python -m agent_system.methods.build_genus_vme_report</code> ·
설명서 <code>GENUS_ISO.md §6</code></p>
</div></body></html>"""

    out = OUT / "genus_vme_risk_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"saved -> {out}")
    print(f"  in-sample VME {agg['insample']['tot_o']}→{agg['insample']['tot_g']} "
          f"(위험 {agg['insample']['n_risk']}, 회귀 {agg['insample']['n_reg']}) | "
          f"OOF VME {agg['oof']['tot_o']}→{agg['oof']['tot_g']} (위험 {agg['oof']['n_risk']})")


if __name__ == "__main__":
    main()
