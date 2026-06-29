#!/usr/bin/env python3
"""워크북(F21_GN Selected model) vs 우리 AUROC-routing 선정 모델 비교 보고서 (HTML).

- 전체 204 cell (워크북 GN ∩ 우리 routing) 일치/불일치 분류
- 일치 tier: 정확일치 / near-tie 관용일치 / 실질 불일치(gap>=0.02)
- 실질 불일치 26 cell 은 운영 SIR/PASS 파이프라인 결과(Task 1) 병합
출력: agent_system/output/workbook_vs_routing/workbook_routing_compare.html
"""
from __future__ import annotations
import html
from pathlib import Path

import pandas as pd

ROOT = Path("/home/kptae/project/qnt_algorithm")
WORKBOOK = ROOT / "claudeCode/data/F21_GN_Selecetd_model_260123 (3) (1).xlsx"
ROUTING_LOOKUP = ROOT / "claudeCode/output_subset_eval/cell_model_routing_lookup.csv"
OUT_DIR = ROOT / "agent_system/output/workbook_vs_routing"
SIR_COMP = OUT_DIR / "comparison_26cells.csv"

NAME_NORM = {
    "deployed_model": "same_mic",
    "object_area_045~08_threshold": "object_area_045~08",
    "object_area_all_threshold": "object_area_all",
}
GAP = 0.02
NEAR_TIE = 0.01


def build_compare() -> pd.DataFrame:
    ours = pd.read_csv(ROUTING_LOOKUP)
    wb = pd.read_excel(WORKBOOK, "Model selected", header=0).dropna(axis=1, how="all")
    wb = wb[["organism_group", "antimicrobial", "Model", "Brightness"]].copy()
    wb = wb.dropna(subset=["organism_group", "Model"])
    wb["brightness"] = wb["Brightness"].fillna(False).astype(bool)
    wb["wb_model"] = wb["Model"].map(lambda m: NAME_NORM.get(str(m).strip(), str(m).strip()))
    for d in (wb, ours):
        d["organism_group"] = d["organism_group"].astype(str).str.strip()
        d["antimicrobial"] = d["antimicrobial"].astype(str).str.strip()
    both = wb.merge(ours, on=["organism_group", "antimicrobial"], how="inner")
    both["runner_up_n"] = both["runner_up"].astype(str).str.replace("_threshold", "", regex=False)
    both["agree"] = both["wb_model"] == both["best_model"]
    both["wb_is_runnerup"] = both["wb_model"] == both["runner_up_n"]

    def tier(r):
        if r["agree"]:
            return "정확 일치"
        if r["wb_is_runnerup"] and r["gap"] < NEAR_TIE:
            return "관용 일치 (near-tie)"
        if r["gap"] >= GAP:
            return "실질 불일치"
        return "경미 불일치"
    both["tier"] = both.apply(tier, axis=1)
    return both.sort_values(["tier", "gap"], ascending=[True, False])


def main() -> None:
    comp = build_compare()
    sir = pd.read_csv(SIR_COMP) if SIR_COMP.exists() else pd.DataFrame()
    sir_map = {}
    if not sir.empty:
        for _, r in sir.iterrows():
            sir_map[(r["organism_group"], r["antimicrobial"])] = r

    n = len(comp)
    tier_counts = comp["tier"].value_counts()
    n_exact = int((comp["tier"] == "정확 일치").sum())
    n_neartie = int((comp["tier"] == "관용 일치 (near-tie)").sum())
    n_real = int((comp["tier"] == "실질 불일치").sum())

    # Brightness(=image-sequence 표현력 낮음) ↔ 불일치 상관
    br = comp["brightness"]
    n_bri = int(br.sum())
    real_mask = comp["tier"] == "실질 불일치"
    bri_in_real = int((real_mask & br).sum())
    real_rate_bri = (real_mask & br).sum() / max(n_bri, 1)
    real_rate_nonbri = (real_mask & ~br).sum() / max((~br).sum(), 1)
    # Brightness=True 에서 우리가 어떤 모델로 가는지
    bri_our = comp.loc[br, "best_model"].value_counts()
    bri_our_str = ", ".join(f"{m} {c}" for m, c in bri_our.items())
    # Brightness=True cell 의 평균 best AUROC vs False
    auroc_bri = comp.loc[br, "best_auroc"].mean()
    auroc_nonbri = comp.loc[~br, "best_auroc"].mean()

    # SIR/PASS summary
    sir_summary = ""
    if not sir.empty:
        our_pass = int((sir["our_FDA_fail_list"] == "PASS").sum())
        wb_pass = int((sir["wb_FDA_fail_list"] == "PASS").sum())
        our_vme = int(sir["our_FDA_fail_list"].astype(str).str.contains("VME").sum())
        wb_vme = int(sir["wb_FDA_fail_list"].astype(str).str.contains("VME").sum())
        sir_summary = f"""
        <div class="cards">
          <div class="card"><div class="big">{our_pass} : {wb_pass}</div>FDA PASS<br><span class="sub">우리 : 워크북</span></div>
          <div class="card warn"><div class="big">{our_vme} : {wb_vme}</div>VME 발생 cell<br><span class="sub">우리 : 워크북 (낮을수록 안전)</span></div>
          <div class="card"><div class="big">21 / 26</div>둘 다 FAIL<br><span class="sub">난이도 높은 cell — 모델 무관</span></div>
        </div>"""

    TIER_COLOR = {
        "정확 일치": "#1a7f37", "관용 일치 (near-tie)": "#0969da",
        "경미 불일치": "#9a6700", "실질 불일치": "#cf222e",
    }

    rows = []
    for _, r in comp.iterrows():
        key = (r["organism_group"], r["antimicrobial"])
        s = sir_map.get(key)
        sir_cells = ""
        if s is not None:
            oc, wc = str(s["our_FDA_fail_list"]), str(s["wb_FDA_fail_list"])
            def badge(v):
                if v == "PASS":
                    return '<span class="pass">PASS</span>'
                cls = "vme" if "VME" in v else "fail"
                return f'<span class="{cls}">{html.escape(v)}</span>'
            sir_cells = f"<td>{badge(oc)}</td><td>{badge(wc)}</td>"
        else:
            sir_cells = '<td class="na">—</td><td class="na">—</td>'
        color = TIER_COLOR.get(r["tier"], "#57606a")
        br = bool(r["brightness"])
        br_cell = ('<span class="bri">True</span>' if br else '<span class="dim">False</span>')
        rows.append(
            f"<tr{' class=brirow' if br else ''}>"
            f"<td>{html.escape(r['organism_group'])}</td>"
            f"<td>{html.escape(r['antimicrobial'])}</td>"
            f"<td style='text-align:center'>{br_cell}</td>"
            f"<td><b>{html.escape(str(r['wb_model']))}</b></td>"
            f"<td><b>{html.escape(str(r['best_model']))}</b></td>"
            f"<td>{html.escape(str(r['runner_up_n']))}</td>"
            f"<td style='text-align:right'>{r['best_auroc']:.4f}</td>"
            f"<td style='text-align:right'>{r['gap']:.4f}</td>"
            f"<td style='color:{color};font-weight:600'>{r['tier']}</td>"
            f"{sir_cells}"
            f"</tr>"
        )

    tier_rows = "".join(
        f"<tr><td style='color:{TIER_COLOR.get(t,'#000')};font-weight:600'>{t}</td>"
        f"<td style='text-align:right'>{c}</td>"
        f"<td style='text-align:right'>{100*c/n:.1f}%</td></tr>"
        for t, c in tier_counts.items()
    )

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>워크북 GN 선정 vs AUROC-routing 비교</title>
<style>
 body{{font-family:-apple-system,'Segoe UI',Roboto,'Malgun Gothic',sans-serif;
   margin:0;padding:28px 36px;color:#1f2328;background:#fff;line-height:1.5;}}
 h1{{font-size:23px;margin:0 0 4px;}} h2{{font-size:17px;margin:26px 0 10px;border-bottom:2px solid #d0d7de;padding-bottom:5px;}}
 .sub{{color:#57606a;font-size:12px;}}
 .cards{{display:flex;gap:14px;margin:14px 0;flex-wrap:wrap;}}
 .card{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:10px;padding:14px 20px;min-width:140px;text-align:center;font-size:12.5px;color:#57606a;}}
 .card.warn{{background:#fff8f0;border-color:#ffb77c;}}
 .card .big{{font-size:26px;font-weight:700;color:#1f2328;margin-bottom:4px;}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0;}}
 th,td{{border:1px solid #d8dee4;padding:5px 8px;}} th{{background:#f0f3f6;text-align:left;position:sticky;top:0;}}
 tr:nth-child(even){{background:#fbfcfd;}}
 .pass{{background:#dafbe1;color:#1a7f37;padding:1px 6px;border-radius:5px;font-weight:600;}}
 .fail{{background:#fff1e5;color:#9a6700;padding:1px 6px;border-radius:5px;}}
 .vme{{background:#ffebe9;color:#cf222e;padding:1px 6px;border-radius:5px;font-weight:600;}}
 .na{{color:#afb8c1;text-align:center;}}
 .note{{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;margin:12px 0;font-size:13px;border-radius:0 6px 6px 0;}}
 .key{{background:#fff8c5;border-left:4px solid #d4a72c;padding:10px 14px;margin:12px 0;font-size:13px;border-radius:0 6px 6px 0;}}
 .bri{{background:#fde7c3;color:#8a5a00;padding:1px 7px;border-radius:5px;font-weight:600;}}
 .dim{{color:#afb8c1;}}
 tr.brirow{{background:#fffaf2;}} tr.brirow:nth-child(even){{background:#fff6e9;}}
</style></head><body>
<h1>워크북 GN 선정 모델 vs AUROC-routing 선정 모델 비교</h1>
<div class="sub">소스: <code>F21_GN_Selecetd_model_260123 (3) (1).xlsx</code> · <code>cell_model_routing_lookup.csv</code> · 교집합 {n} cell · 2026-06-02</div>

<div class="note">
<b>두 시스템의 선정 기준이 다름.</b> 워크북 <code>Model selected</code> = <b>FDA 임상 metric(EA/CA/VME/PASS)</b> 기준,
우리 routing = <b>순수 best AUROC(판별력)</b> 기준. 모델명은 정규화됨 (<code>deployed_model→same_mic</code>,
<code>object_area_*_threshold→object_area_*</code>).
</div>

<h2>1. 일치도 요약 (교집합 {n} cell)</h2>
<table style="max-width:520px">
<tr><th>분류</th><th style="text-align:right">cell 수</th><th style="text-align:right">비율</th></tr>
{tier_rows}
</table>
<div class="key">
정확 일치는 <b>{n_exact} ({100*n_exact/n:.1f}%)</b>로 낮지만, 불일치 대부분은 1·2위 AUROC차(gap)가 미미한
<b>near-tie</b>다. 정확+관용 일치 = <b>{n_exact+n_neartie} ({100*(n_exact+n_neartie)/n:.1f}%)</b>.
판별력에 의미있는 차이(gap≥{GAP})가 나는 <b>실질 불일치는 {n_real} cell ({100*n_real/n:.1f}%)</b>뿐이며,
대부분 <b>carbapenem(MP/ETP/IP)·colistin(CL)·cephalosporin × Enterobacterales</b>에 집중된다.
</div>

<h2>2. 실질 불일치 26 cell — 운영 SIR/PASS 파이프라인 비교 (Task 1)</h2>
<div class="sub">운영 recipe와 동일: raw model_pred → per-cell isotonic(FDA fit) → threshold 0.65 → drast_gng→MIC→SIR→evalEA</div>
{sir_summary}
<div class="key">
<b>결론: 충돌 cell에서 우리 AUROC 선택이 임상적으로 동등하거나 더 안전.</b>
FDA PASS는 우리 5 vs 워크북 4 (1 cell 우리 우위, 나머지 동률). 결정적으로 <b>VME 발생은 우리 6 cell vs 워크북 10 cell</b>이며,
워크북 선택이 VME를 유발하나 우리 선택은 깨끗한 cell이 <b>4건</b>(반대 0건). 즉 우리 <code>gram_negative</code>/<code>beta_lactam</code>
선택이 워크북의 <code>same_mic</code>/<code>object_area</code>보다 VME 측면에서 더 안전하다.
단 26 cell 중 21개는 <b>모델 무관하게 둘 다 FAIL</b>인 본질적 난이도 cell(carbapenem/colistin)이다.
</div>

<h2>3. Brightness 플래그 분석 (image sequence 표현력 낮은 조합)</h2>
<div class="note">
워크북의 <code>Brightness=True</code>는 <b>image sequence(시계열 이미지)의 표현력이 떨어지는 조합</b>으로,
이미지 기반 <code>model_pred</code>의 신뢰도가 낮아 brightness/object_area 등 보조 feature가 더 중요해지는 cell이다.
교집합 {n} cell 중 <b>{n_bri}개({100*n_bri/n:.0f}%)가 Brightness=True</b>.
</div>
<table style="max-width:640px">
<tr><th>지표</th><th style="text-align:right">Brightness=True</th><th style="text-align:right">Brightness=False</th></tr>
<tr><td>cell 수</td><td style="text-align:right">{n_bri}</td><td style="text-align:right">{n-n_bri}</td></tr>
<tr><td>실질 불일치율 (gap≥{GAP})</td><td style="text-align:right;color:#cf222e;font-weight:600">{100*real_rate_bri:.1f}%</td><td style="text-align:right">{100*real_rate_nonbri:.1f}%</td></tr>
<tr><td>평균 best AUROC</td><td style="text-align:right">{auroc_bri:.4f}</td><td style="text-align:right">{auroc_nonbri:.4f}</td></tr>
</table>
<div class="key">
<b>Brightness=True cell에서 실질 불일치가 집중된다</b> — 불일치율 {100*real_rate_bri:.1f}% vs False {100*real_rate_nonbri:.1f}%
(실질 불일치 26 cell 중 <b>{bri_in_real}개가 Brightness=True</b>). image sequence 표현력이 약하니
판별력 최상위 모델이 cell마다 흔들리고(평균 best AUROC도 {auroc_bri:.3f} &lt; {auroc_nonbri:.3f}로 낮음),
그래서 워크북(임상 기준)과 우리(AUROC 기준)의 선택이 갈리는 것. Brightness=True에서 우리 선택 분포:
<b>{bri_our_str}</b> — <code>gram_negative</code>·<code>object_area</code> 계열로 쏠리며,
이는 "이미지가 약할 때 보조 feature 의존" 패턴과 일치한다.
</div>

<h2>4. 전체 {n} cell 비교표</h2>
<div class="sub">SIR 컬럼은 실질 불일치 26 cell에만 표시. <span class="pass">PASS</span> /
<span class="fail">FAIL(비VME)</span> / <span class="vme">VME 포함 FAIL</span></div>
<table>
<tr><th>organism_group</th><th>antimicrobial</th><th>Brightness</th><th>워크북 선정</th><th>우리 선정(best)</th>
<th>우리 runner-up</th><th style="text-align:right">best AUROC</th><th style="text-align:right">gap</th>
<th>분류</th><th>우리 SIR</th><th>워크북 SIR</th></tr>
{''.join(rows)}
</table>
</body></html>"""

    out = OUT_DIR / "workbook_routing_compare.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[saved] {out}  ({len(doc)/1024:.1f} KB, {n} cells)")


if __name__ == "__main__":
    main()
