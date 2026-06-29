"""methods 패키지 카탈로그·검증 HTML 보고서 생성 (자체완결형).

내용: 4-스테이지 파이프라인 · 방법론 10종 카탈로그(중요도/스테이지/상태) · 단일 통합 입력 설계
· Recipe 조립 예시 · 교차검증 4/4 · ens_rankavg(GBM) held-out 재기각.

출력: agent_system/methods/methods_catalog_report.html
재현: python -m agent_system.methods.build_methods_report
"""
from __future__ import annotations

from agent_system.methods import catalog, fixes, base as B

# 교차검증 결과 (validate_recipes.py 로 재현; 각 행은 독립 기준)
VALIDATION = [
    ("routed + iso(insample) @0.65", 103, 103, "운영 routed_iso_t65 (배포 수치)"),
    ("routed + iso(oof) @0.65", 87, 87, "held-out (2026-06-04 brightness CV baseline)"),
    ("rankavg_w0.3 + iso(oof) @0.65", 83, 83, "ens_rankavg(GBM) held-out (−4)"),
    ("brightness + iso(insample) @0.65", 106, 106, "in-sample +3 (2026-06-02)"),
]

# (b) ens_rankavg(GBM) 재검토 (rankavg_gbm_iso_cv.py, threshold 0.65)
RANKAVG = [
    ("routed (control)", 103, 87, "—"),
    ("rankavg w0.3 (GBM 30%)", 103, 83, "−4"),
    ("rankavg w0.5 (GBM 50%)", 100, 78, "−9"),
    ("gbm_only", 78, 56, "−31"),
]

STAGE_LABEL = {"score": "A · SCORE", "calibrate": "B · CALIBRATE",
               "threshold": "C · THRESHOLD", "evaluate": "D · EVALUATE"}
STATUS_BADGE = {
    "adopted": ("✅ 채택", "#1b5e20", "#e8f5e9"),
    "auxiliary": ("🟡 보조", "#e65100", "#fff3e0"),
    "alternative": ("🟡 대안", "#6a1b9a", "#f3e5f5"),
    "rejected": ("🔴 기각", "#b71c1c", "#ffebee"),
}


def _rows() -> str:
    out = ""
    for i, m in enumerate(catalog(), 1):
        badge, fg, bg = STATUS_BADGE.get(m["status"], (m["status"], "#333", "#eee"))
        star = ' <span class="lnz">★LNZ</span>' if "★LNZ" in m["pass_impact"] else ""
        out += (
            f'<tr><td class="num">{i}</td>'
            f'<td><code>{m["class"]}</code>{star}<div class="sm">{m["summary"]}</div></td>'
            f'<td>{STAGE_LABEL.get(m["stage"], m["stage"])}</td>'
            f'<td><span class="badge" style="color:{fg};background:{bg}">{badge}</span></td>'
            f'<td class="imp">{m["pass_impact"]}</td></tr>\n')
    return out


def _fix_rows() -> str:
    out = ""
    for m in fixes():
        out += (f'<tr><td><code>{m["class"]}</code><div class="sm">{m["summary"]}</div></td>'
                f'<td>{STAGE_LABEL.get(m["stage"], m["stage"])}</td>'
                f'<td class="imp">{m["pass_impact"]}</td></tr>\n')
    return out


def _validation_rows() -> str:
    out = ""
    for label, expect, got, note in VALIDATION:
        ok = "✓" if expect == got else "✗"
        cls = "ok" if expect == got else "bad"
        out += (f'<tr><td>{label}</td><td class="c">{expect}</td>'
                f'<td class="c {cls}">{got} {ok}</td><td class="sm">{note}</td></tr>\n')
    return out


def _rankavg_rows() -> str:
    out = ""
    for label, ins, oof, d in RANKAVG:
        hl = ' class="ctrl"' if "control" in label else ""
        out += (f'<tr{hl}><td>{label}</td><td class="c">{ins}</td>'
                f'<td class="c">{oof}</td><td class="c d">{d}</td></tr>\n')
    return out


def build_html() -> str:
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>methods 패키지 — 방법론 카탈로그 · 검증</title>
<style>
 :root{{--blue:#2979ff;--ink:#222}}
 *{{box-sizing:border-box}}
 body{{font-family:-apple-system,system-ui,'Malgun Gothic',sans-serif;background:#f4f6f9;
   margin:0;color:var(--ink);line-height:1.6}}
 .wrap{{max-width:1080px;margin:0 auto;padding:28px 30px 90px}}
 h1{{font-size:24px;border-bottom:3px solid var(--blue);padding-bottom:10px;margin-bottom:4px}}
 .sub{{color:#666;font-size:13px;margin:6px 0 18px}}
 h2{{font-size:18px;color:#1565c0;margin:32px 0 8px;border-left:5px solid var(--blue);padding-left:11px}}
 .ops{{background:#e8f5e9;border:1px solid #4caf50;border-radius:8px;padding:14px 18px;margin:16px 0;font-size:13.5px}}
 .ops b{{color:#1b5e20}}
 .flow{{background:#0d2440;color:#cfe3ff;border-radius:9px;padding:16px 18px;font-family:ui-monospace,monospace;
   font-size:13px;overflow-x:auto;white-space:pre;margin:10px 0}}
 table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff;border-radius:8px;overflow:hidden;
   box-shadow:0 1px 3px rgba(0,0,0,.06);margin:8px 0}}
 th,td{{border:1px solid #e3e7ee;padding:7px 10px;text-align:left;vertical-align:top}}
 th{{background:#eef4ff;color:#0d47a1;font-size:12.5px}}
 td.num,td.c{{text-align:center;white-space:nowrap}}
 td.imp,td.d{{font-weight:700;color:#1565c0;white-space:nowrap}}
 .sm{{font-size:11.5px;color:#555;margin-top:3px}}
 .badge{{font-size:11.5px;border-radius:5px;padding:2px 8px;white-space:nowrap;font-weight:600}}
 .lnz{{font-size:10.5px;background:#d84315;color:#fff;border-radius:4px;padding:1px 6px}}
 code{{background:#eceff3;padding:1px 6px;border-radius:4px;font-size:12px}}
 pre{{background:#1e1e2e;color:#dce3f0;border-radius:8px;padding:14px 16px;overflow-x:auto;font-size:12.5px}}
 pre .k{{color:#82aaff}} pre .s{{color:#c3e88d}} pre .c{{color:#7e8aa3}}
 td.ok{{color:#1b5e20;font-weight:700}} td.bad{{color:#b71c1c;font-weight:700}}
 tr.ctrl{{background:#fffde7}}
 .note{{font-size:12.5px;color:#555;background:#f7f9fc;border:1px solid #e3e7ee;border-radius:7px;padding:10px 14px;margin:8px 0}}
 footer{{color:#888;font-size:11.5px;margin-top:36px;border-top:1px solid #ddd;padding-top:12px}}
</style></head><body><div class="wrap">
<h1>methods 패키지 — FDA PASS 방법론 카탈로그 · 검증</h1>
<p class="sub">QuantaMatrix dRAST 3.0 · <code>agent_system/methods/</code> · FDA2023 평가셋 · 방법론 9종을
4-스테이지 공통 인터페이스로 모듈화 (+ 평가기 버그수정 1종 별도) · 입력은 단일 통합 파일</p>

<div class="ops">
<b>운영 권고 — <code>routed_iso_t65</code></b> = PerCellModelRouting(★LNZ) + PerCellIsotonic + threshold 0.65 ·
FDA PASS <b>103</b>/313 · VME-rate <b>1.99%</b> · AUROC <b>0.9743</b>.
&nbsp; <code>operational_recipe().evaluate()</code> 한 줄로 재현.
</div>

<h2>1. 4-스테이지 파이프라인</h2>
<div class="flow">per_conc ─▶ [A.SCORE] ─▶ [B.CALIBRATE] ─▶ [C.THRESHOLD] ─▶ [D.EVALUATE] ─▶ PASS/VME
           점수 선택/변환    score→P(NG)        P(NG)→G/NG        MIC→SIR→EA</div>
<p class="note">모든 SCORE/CALIBRATE 방법론은 동일 계약 <code>apply(per_conc) → per_conc</code>(활성 점수
<code>ng_score</code> 갱신)을 따른다. <code>Recipe(score, calibrate, threshold)</code>로 조립하면
운영 <code>verify_method_sir_pipeline</code>(EA fix 내장)이 MIC→SIR→EA/CA/ME/VME→PASS 를 산출.</p>

<h2>2. 방법론 카탈로그 (중요도 = FDA PASS 기여 순)</h2>
<table>
<thead><tr><th>#</th><th>방법론 / 요약</th><th>스테이지</th><th>상태</th><th>PASS 기여</th></tr></thead>
<tbody>
{_rows()}</tbody></table>
<p class="note"><b>★ LNZ(linezolid) 통과 방법론 = <code>PerCellModelRouting</code></b> — baseline(same_mic)은 LNZ
gram-positive 5 cell 전부 FAIL. cell별 best subset 모델 선택(E. faecium→early_stop,
S. Coag-neg→beta_lactam, E. faecalis→object_area_045~08, S. lugdunensis→early_stop) → <b>4/5 PASS</b>.</p>

<h2>2-1. 평가기 버그 수정 (방법론 아님)</h2>
<p class="note">아래는 scoring/calibration <b>방법론이 아니라 평가기 correctness 수정</b>이다. PASS 를
새로 끌어올리는 개선책이 아니라, BMD↔dRAST notation mismatch 로 <b>잘못 FAIL 되던 cell 을 정정</b>한다
(정정 효과 71→103, 원본 True 보존·PASS→FAIL=0). 파이프라인 필수 컴포넌트로 verify 가 자동 적용.</p>
<table>
<thead><tr><th>버그 수정 / 요약</th><th>스테이지</th><th>효과</th></tr></thead>
<tbody>
{_fix_rows()}</tbody></table>

<h2>3. 단일 통합 입력</h2>
<p class="note">흩어진 per-row CSV 를 <code>build_unified_input.py</code> 가 하나의 통합 per_conc
(<code>data/unified_per_conc.parquet</code>, 152,834행)로 합친다. 각 방법론은 외부 파일을 따로 읽지 않고
아래 source 컬럼만 소비한다.</p>
<table>
<thead><tr><th>컬럼</th><th>의미</th></tr></thead><tbody>
<tr><td><code>src_same_mic</code></td><td>baseline same_mic.pt 점수</td></tr>
<tr><td><code>src_routed</code></td><td>per-cell routed model_pred (★LNZ)</td></tr>
<tr><td><code>src_objarea_gbm</code></td><td>object_area cross-domain GBM 예측</td></tr>
<tr><td><code>src_object_area</code></td><td>raw object_area DTW</td></tr>
<tr><td><code>src_brightness_gbm</code></td><td>brightness GBM 예측</td></tr>
<tr><td><code>bucket</code></td><td>aligned / shifted / fda_only (domain shift)</td></tr>
<tr><td><code>ng_score</code></td><td><b>활성 점수</b> (SCORE/CALIBRATE 가 갱신; 외부 dtw_model_pred 와 동일 의미, DTW 아님)</td></tr>
</tbody></table>

<h2>4. Recipe 조립 (운영 재현)</h2>
<pre><span class="k">from</span> agent_system.methods <span class="k">import</span> operational_recipe
operational_recipe().evaluate()        <span class="c"># routed_iso_t65 → {{'pass':103,...}}</span>

<span class="c"># 직접 조립</span>
<span class="k">from</span> agent_system.methods.recipe <span class="k">import</span> Recipe
<span class="k">from</span> agent_system.methods.score.model_routing <span class="k">import</span> PerCellModelRouting
<span class="k">from</span> agent_system.methods.calibrate.isotonic <span class="k">import</span> PerCellIsotonic
Recipe(score=PerCellModelRouting(), calibrate=PerCellIsotonic(<span class="s">"insample"</span>),
       threshold=<span class="s">0.65</span>).evaluate()</pre>

<h2>5. 교차검증 — 모듈 조립이 독립 기준 재현 (4/4 ✓)</h2>
<table>
<thead><tr><th>레시피 (모듈 조립)</th><th>기대</th><th>산출</th><th>기준 출처</th></tr></thead>
<tbody>
{_validation_rows()}</tbody></table>
<p class="note">기존 검증 클래스를 thin-wrap 했으므로 로직 재구현·회귀 없음. 재현:
<code>python -m agent_system.methods.validate_recipes</code>.</p>

<h2>6. ens_rankavg(GBM) 운영 경로 재검토 — 기각 (2026-06-05)</h2>
<p class="note">사용자 질의: "ens_rankavg 에 GBM 이 들어있는데 운영에 살릴 수 있나?" → rank-scale per-cell ISO
재calibrate + GroupKFold(sample_id) <b>held-out OOF</b> + EA-fix + 전체 cell 로 정직하게 재평가
(<code>rankavg_gbm_iso_cv.py</code>, threshold 0.65).</p>
<table>
<thead><tr><th>후보</th><th>in-sample</th><th>OOF (held-out)</th><th>ΔOOF vs routed</th></tr></thead>
<tbody>
{_rankavg_rows()}</tbody></table>
<p class="note"><b>판정</b>: GBM 을 rank-avg 로 섞으면 held-out PASS 가 개선되지 않고 <b>악화</b>(−4~−9).
cell-level w0.3 = 14 gain / 18 loss(net −4) — AUROC 는 오르나 PASS 손해(축 분리). Cycle 2
"rank_avg 기여 0" 의 더 엄밀한 확정 → 운영 <code>routed_iso_t65</code> 불변, GBM 은 AUROC-only 자산.</p>

<footer>생성: <code>python -m agent_system.methods.build_methods_report</code> ·
패키지 README: <code>agent_system/methods/README.md</code></footer>
</div></body></html>"""


def main() -> None:
    out = B.PKG_DIR / "methods_catalog_report.html"
    out.write_text(build_html(), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
