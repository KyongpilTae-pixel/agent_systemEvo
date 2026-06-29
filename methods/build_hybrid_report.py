"""Hybrid isotonic 비교 → 단일 HTML 보고서.

organism / genus / hybrid 를 in-sample·OOF 로 비교(PASS·VME-cell·총 VME) + fallback cell 목록
+ (있으면) threshold sweep. 기존 평가 산출 dir(hyb_<variant>_<mode>, hybsw_*)를 읽어 재현.

전제: hybrid_iso_eval.py 실행(6 dir 생성). threshold sweep 은 hybrid_threshold_sweep.py(선택).
사용: python -m agent_system.methods.build_hybrid_report
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B

OUT = B.PKG_DIR / "output"
KEY = ["organism_group", "antimicrobial"]


def _metrics(dirname):
    d = OUT / dirname
    sm = pd.read_csv(d / "method_summary_model_pred.csv")
    ev = pd.read_csv(d / "method_eval_model_pred.csv")
    n_pass = int((sm["FDA_fail_list"] == "PASS").sum())
    n_vmec = int(sm["FDA_fail_list"].astype(str).str.contains("VME").sum())
    tot_vme = int(ev["isVME"].sum())
    try:
        feas = pd.read_csv(d / "method_sir_feasibility.csv")
        vr = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    except Exception:
        vr = float("nan")
    return dict(pass_=n_pass, vme_cell=n_vmec, tot_vme=tot_vme, vme_rate=vr)


def _variant_table() -> str:
    rows = ""
    best = {}
    data = {}
    for mode in ("insample", "oof"):
        for v in ("organism", "genus", "hybrid"):
            data[(v, mode)] = _metrics(f"hyb_{v}_{mode}")
    for mode in ("insample", "oof"):
        # best per mode for highlight
        ps = {v: data[(v, mode)]["pass_"] for v in ("organism", "genus", "hybrid")}
        vs = {v: data[(v, mode)]["tot_vme"] for v in ("organism", "genus", "hybrid")}
        best_pass = max(ps.values()); best_vme = min(vs.values())
        for v in ("organism", "genus", "hybrid"):
            m = data[(v, mode)]
            hl = ' class="hyb"' if v == "hybrid" else ""
            pc = ' class="best"' if m["pass_"] == best_pass else ""
            vc = ' class="best"' if m["tot_vme"] == best_vme else ""
            label = {"organism": "organism (운영)", "genus": "genus", "hybrid": "hybrid ★"}[v]
            md = "in-sample" if mode == "insample" else "OOF (held-out)"
            rows += (f"<tr{hl}><td>{md}</td><td>{label}</td>"
                     f"<td{pc}>{m['pass_']}</td><td>{m['vme_cell']}</td>"
                     f"<td{vc}>{m['tot_vme']}</td></tr>\n")
    return rows


def _sweep_table() -> str:
    """hybrid threshold sweep (hybsw_<mode>_t<thr>) — 있으면 렌더, 없으면 빈 문자열."""
    import glob
    dirs = sorted(glob.glob(str(OUT / "hybsw_*")))
    if not dirs:
        return ""
    rows = ""
    for d in dirs:
        name = d.split("/")[-1]                      # hybsw_oof_t65
        _, mode, t = name.split("_")
        thr = int(t[1:]) / 100
        m = _metrics(name)
        rows += (f"<tr><td>{'OOF' if mode=='oof' else 'in-sample'}</td><td>{thr:.2f}</td>"
                 f"<td>{m['pass_']}</td><td>{m['vme_cell']}</td><td>{m['tot_vme']}</td>"
                 f"<td>{m['vme_rate']*100:.2f}%</td></tr>\n")
    return f"""
<h2>3. Hybrid threshold sweep</h2>
<p class="note">hybrid 의 threshold(0.50/0.65/0.75) × VME budget. 운영 작동점 선정용.</p>
<table><thead><tr><th>mode</th><th>threshold</th><th>PASS</th><th>VME-cell</th>
<th>총 VME</th><th>VME-rate</th></tr></thead><tbody>{rows}</tbody></table>
"""


def main() -> None:
    fb = pd.read_csv(OUT / "hybrid_fallback_cells.csv")
    fb_rows = "".join(f"<tr><td>{r[B.OG_COL]}</td><td>{r[B.DRUG_COL]}</td></tr>"
                      for _, r in fb.iterrows())

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hybrid isotonic — genus + VME 위험 cell organism 폴백</title>
<style>
 body{{font-family:-apple-system,system-ui,'Malgun Gothic',sans-serif;background:#f4f6f9;margin:0;color:#222;line-height:1.6}}
 .wrap{{max-width:980px;margin:0 auto;padding:26px 30px 80px}}
 h1{{font-size:22px;border-bottom:3px solid #2e7d32;padding-bottom:9px}}
 h2{{font-size:17px;color:#1b5e20;margin:28px 0 8px;border-left:5px solid #2e7d32;padding-left:11px}}
 .sub{{color:#666;font-size:13px;margin:6px 0 14px}}
 .win{{background:#e8f5e9;border:1px solid #2e7d32;border-radius:8px;padding:13px 17px;margin:14px 0;font-size:13.5px}}
 .win b{{color:#1b5e20}}
 table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06);margin:8px 0}}
 th,td{{border:1px solid #e3e7ee;padding:7px 10px;text-align:center}}
 th{{background:#e8f5e9;color:#1b5e20}}
 tr.hyb{{background:#f1f8e9;font-weight:600}}
 td.best{{background:#c8e6c9;font-weight:700}}
 .note{{font-size:12.5px;color:#555;background:#f7f9fc;border:1px solid #e3e7ee;border-radius:7px;padding:10px 14px;margin:8px 0}}
 .two{{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}}
 .two .col{{flex:1;min-width:300px}}
 code{{background:#eceff3;padding:1px 6px;border-radius:4px}}
</style></head><body><div class="wrap">
<h1>Hybrid isotonic — genus 기본 + VME 위험 cell 만 organism 폴백</h1>
<p class="sub">routed model_pred → per-cell isotonic(hybrid) → threshold 0.65 ·
학습/적용 genus, VME 위험 7 cell 만 organism · summary = organism_group</p>

<div class="win">
<b>결론</b>: hybrid 가 <b>두 축(PASS·VME) 모두 organism·genus 를 능가</b>. held-out PASS 는 genus 의
일반화 이득(+4) 유지(OOF 87→91), 총 VME 는 두 모드 모두 최저(OOF 97/92→<b>87</b>). genus 의 VME
감소를 살리되 7 cell 의 VME 회귀만 organism 으로 차단.
</div>

<h2>1. 비교 — organism vs genus vs hybrid</h2>
<table><thead><tr><th>mode</th><th>variant</th><th>PASS</th><th>VME-cell</th>
<th>총 VME isolate</th></tr></thead><tbody>{_variant_table()}</tbody></table>
<p class="note">초록 셀 = 그 mode 의 최선(PASS 최대 / 총 VME 최소). hybrid 행 강조. OOF 가 정직한
일반화 추정(학습/평가 분리, GroupKFold sample_id).</p>

<div class="two">
<div class="col">
<h2>2. Fallback cell ({len(fb)}개)</h2>
<p class="note">in-sample 에서 genus VME &gt; organism VME 인 cell — 이 cell 만 organism-iso 유지.
배포 자산 <code>hybrid_fallback_cells.csv</code>.</p>
<table><thead><tr><th>organism_group</th><th>약제</th></tr></thead><tbody>{fb_rows}</tbody></table>
</div>
<div class="col">
<h2>메커니즘</h2>
<div class="note">
<ul>
<li>genus-iso 는 <b>대부분 cell 에서 VME 감소</b>(표본 큰 곡선이 덜 overfit).</li>
<li>단 S.aureus·Proteus 등 7 cell 은 species 차이를 뭉개 <b>R→S(VME)</b> 발생.</li>
<li>그 7 cell 만 organism 으로 되돌리면 <b>genus 이득 + VME 회귀 차단</b> 동시 달성.</li>
</ul></div>
</div>
</div>
{_sweep_table()}

<h2>운영 함의 · 다음</h2>
<div class="note">
<ol>
<li>hybrid 는 운영 organism(routed_iso_t65) 대비 <b>held-out PASS +4 AND VME −10</b> — 더 통과하며
더 안전. 정직한 OOF 기준 개선.</li>
<li>fallback set(7 cell)은 in-sample VME 비교 도출. 운영 채택 전 fold 안정성 + threshold×VME budget
확인(§3 sweep).</li>
<li>구현 <code>calibrate/hybrid_iso.py::HybridIsotonic</code> · 평가 <code>hybrid_iso_eval.py</code>.</li>
</ol></div>
<p style="color:#888;font-size:11.5px;margin-top:22px">생성:
<code>python -m agent_system.methods.build_hybrid_report</code></p>
</div></body></html>"""

    out = OUT / "hybrid_iso_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
