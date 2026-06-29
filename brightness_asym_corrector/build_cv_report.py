#!/usr/bin/env python3
"""GroupKFold(sample_id) CV 결과 보고서 HTML 생성.

groupkfold_cv.py 의 산출(artifacts/cv/cv_manifest.json + 각 SIR summary)을 읽어
held-out 검증 결과를 한 장으로 정리. 출력: artifacts/cv/cv_report.html
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parent
CV = PKG / "artifacts" / "cv"


def counts(d: str) -> tuple[int, int, float]:
    s = pd.read_csv(CV / d / "method_summary_model_pred.csv")
    p = int((s["FDA_fail_list"] == "PASS").sum())
    v = int(s["FDA_fail_list"].astype(str).str.contains("VME").sum())
    feas = pd.read_csv(CV / d / "method_sir_feasibility.csv")
    vr = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    return p, v, vr


def main() -> None:
    man = json.loads((CV / "cv_manifest.json").read_text())
    bi_p, bi_v, bi_r = counts("baseline_insample")
    ci_p, ci_v, ci_r = counts("corrected_insample")
    bo_p, bo_v, bo_r = counts("baseline_oof")
    weights = [w for w in (0.5, 0.7, 1.0) if (CV / f"corrected_oof_w{w}").exists()]
    oof = {w: counts(f"corrected_oof_w{w}") for w in weights}
    co_p, co_v, co_r = oof[0.7] if 0.7 in oof else list(oof.values())[0]

    loss = pd.read_csv(CV / "cv_compare.csv")
    loss_rows = loss[loss["base_oof_pass"] & (~loss["corr_oof_pass"])]
    gain_rows = loss[(~loss["base_oof_pass"]) & loss["corr_oof_pass"]]

    def cell(v, ref=None, inv=False, suf=""):
        if ref is None:
            return f'<td style="text-align:right">{v}{suf}</td>'
        d = v - ref
        good = (d < 0) if inv else (d > 0)
        col = "#2e7d32" if d == 0 else ("#2e7d32" if good else "#c62828")
        sign = "—" if d == 0 else f"{d:+d}" if isinstance(d, int) else f"{d:+.4f}"
        return (f'<td style="text-align:right">{v}{suf} '
                f'<span style="color:{col};font-size:11px">({sign})</span></td>')

    wsweep = ""
    for w in weights:
        p, v, r = oof[w]
        wsweep += (f"<tr><td>w = {w}</td>{cell(p, bo_p)}{cell(v, bo_v, inv=True)}"
                   f'<td style="text-align:right">{r*100:.2f}%</td></tr>')

    loss_html = "".join(
        f"<tr><td>{r.organism_group} × {r.antimicrobial}</td>"
        f"<td>{r.base_oof_fail}</td><td style='color:#c62828'>{r.corr_oof_fail}</td></tr>"
        for r in loss_rows.itertuples()) or \
        "<tr><td colspan=3 style='text-align:center;color:#888'>없음</td></tr>"
    gain_html = "".join(
        f"<tr><td>{r.organism_group} × {r.antimicrobial}</td>"
        f"<td>{r.base_oof_fail}</td><td style='color:#2e7d32'>{r.corr_oof_fail}</td></tr>"
        for r in gain_rows.itertuples()) or \
        "<tr><td colspan=3 style='text-align:center;color:#888'>없음 (held-out PASS gain 0)</td></tr>"

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>Brightness Corrector — GroupKFold CV 검증</title>
<style>
 body{{font-family:-apple-system,system-ui,'Malgun Gothic',sans-serif;background:#fafafa;margin:0;color:#222;line-height:1.6}}
 .c{{max-width:980px;margin:0 auto;padding:26px 32px 80px}}
 h1{{font-size:22px;border-bottom:3px solid #c62828;padding-bottom:8px}}
 h2{{font-size:16px;color:#1565c0;margin-top:28px;border-left:4px solid #2979ff;padding-left:10px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}}
 th,td{{border:1px solid #ddd;padding:6px 10px}} th{{background:#eef2f7;text-align:right}}
 td:first-child,th:first-child{{text-align:left;font-weight:600}}
 .verdict{{background:#ffebee;border:2px solid #c62828;border-radius:8px;padding:16px 20px;margin:16px 0;font-size:14px}}
 .verdict b{{color:#b71c1c}}
 .key{{background:#fff8e1;border-left:4px solid #f9a825;padding:12px 16px;margin-top:14px;border-radius:0 6px 6px 0;font-size:13px}}
 .note{{color:#777;font-size:11.5px;margin-top:6px}}
 code{{background:#eceff3;padding:1px 6px;border-radius:4px;font-size:12px}}
 .hl{{background:#fff3e0}}
</style></head><body><div class="c">
<h1>Brightness Asymmetric Corrector — GroupKFold(sample_id) CV 검증</h1>
<p>운영 검증(evaluate.py)은 per-cell isotonic 을 <b>전체 FDA 패널에 in-sample fit</b> 하고
brightness 비중 <code>w=0.7</code> 도 FDA PASS 를 보고 선택했다. 그 <b>+3 PASS 가 held-out 에서도
유지되는지</b> GroupKFold(sample_id, {man['folds']}-fold) OOF 로 검증.</p>

<div class="verdict">
<b>결론: +3 PASS 는 held-out 에서 유지되지 않는다 (CV 실패).</b><br>
in-sample baseline <b>{bi_p}</b> → corrected <b>{ci_p}</b> (Δ+{ci_p-bi_p}) 이지만,
held-out OOF 에서는 baseline <b>{bo_p}</b> → corrected <b>{co_p}</b> (<b style="color:#b71c1c">Δ{co_p-bo_p:+d}</b>).
held-out PASS gain cell <b>0</b> / loss cell <b>{len(loss_rows)}</b>, VME-cell {bo_v}→{co_v}.
모든 w 에서 corrected ≤ baseline → +3 은 <b>in-sample 과적합</b>(isotonic 재적합 + w 선택)의 산물.
</div>

<h2>설계</h2>
<p>FDA 에서 in-sample 으로 학습되는 유일한 단계 = <b>per-cell isotonic</b>. (brightness GBM 은
학습 CSV로 out-of-domain 학습 → FDA 에 in-sample 아님.) 따라서 GroupKFold OOF 가 그 단계를 정확히 타격:
각 fold 에서 train sample 로 isotonic fit → held-out sample 만 calibrate, 5 fold 의 held-out 예측을
이어붙여 전체 패널을 재구성(누설 0) 후 운영 SIR/PASS 파이프라인 1회 실행. brightness 보정(w)은
isotonic 이전 단계라 결정적으로 적용.</p>

<h2>핵심 결과 — in-sample vs held-out (313 cell, threshold 0.65)</h2>
<table>
<thead><tr><th>조건</th><th>PASS</th><th>VME-cell</th><th>VME-rate</th></tr></thead>
<tbody>
<tr><td>baseline — in-sample (운영 reference)</td>{cell(bi_p)}{cell(bi_v)}<td style="text-align:right">{bi_r*100:.2f}%</td></tr>
<tr><td>corrected — in-sample (w=0.7, 운영 reference)</td>{cell(ci_p, bi_p)}{cell(ci_v, bi_v, inv=True)}<td style="text-align:right">{ci_r*100:.2f}%</td></tr>
<tr class="hl"><td><b>baseline — OOF (held-out)</b></td>{cell(bo_p)}{cell(bo_v)}<td style="text-align:right">{bo_r*100:.2f}%</td></tr>
<tr class="hl"><td><b>corrected — OOF (held-out, w=0.7)</b></td>{cell(co_p, bo_p)}{cell(co_v, bo_v, inv=True)}<td style="text-align:right">{co_r*100:.2f}%</td></tr>
</tbody></table>
<p class="note">괄호 = 직전 baseline 대비 Δ. in-sample 은 같은 데이터로 fit/평가, OOF 는 자신을 못 본
isotonic 으로만 보정한 held-out 추정.</p>

<h2>w-민감도 (held-out OOF)</h2>
<table>
<thead><tr><th>brightness 비중 w</th><th>PASS (vs baseline OOF {bo_p})</th><th>VME-cell (vs {bo_v})</th><th>VME-rate</th></tr></thead>
<tbody>{wsweep}</tbody></table>
<p class="note">어떤 w 에서도 held-out PASS 가 baseline OOF 를 넘지 못함 — w=0.7 이 cherry-pick 이라기보다
<b>brightness 보정 자체가 held-out 이득이 없다</b>. w 가 클수록(1.0) 오히려 악화.</p>

<h2>held-out cell 변화</h2>
<table>
<thead><tr><th>PASS gain cell (baseline FAIL → corrected PASS)</th><th>baseline</th><th>corrected</th></tr></thead>
<tbody>{gain_html}</tbody></table>
<table style="margin-top:10px">
<thead><tr><th>PASS loss cell (baseline PASS → corrected FAIL)</th><th>baseline</th><th>corrected</th></tr></thead>
<tbody>{loss_html}</tbody></table>

<div class="key">
<b>해석.</b> in-sample 에서 isotonic 은 보정된 score 와 라벨 관계를 <b>해당 cell 안에서 직접
기억</b>해 3개 cephalosporin cell 을 PASS 문턱 위로 올린다. held-out isotonic 은 그 cell 의 보정 score 를
본 적이 없어 이득이 사라지고, G-방향 보정이 오히려 일부 cell 에서 VME 를 늘림(VME-cell {bo_v}→{co_v}).
부수적으로 <b>운영 isotonic 자체의 in-sample 낙관</b>도 정량화됨: baseline 103(in-sample) → 87(held-out).
운영 배포는 관례적으로 전체 데이터 isotonic fit 을 쓰므로 103 은 배포 수치, 87 은 미관측 sample 일반화 추정.
</div>

<h2>권고</h2>
<div class="verdict" style="background:#fff;border-color:#1565c0">
<b style="color:#0d47a1">brightness 보정을 운영에 투입하지 않는다.</b> +3 PASS 는 held-out 에서 재현되지 않음.
운영 권고는 <code>routed_iso_t65</code> (in-sample PASS 103) 유지. 본 모듈은 <b>in-sample only /
CV 실패</b>로 표기. (향후: cell 내부가 아닌 cross-domain 학습형 보정자, 또는 brightness 를
isotonic 이전이 아닌 라벨-독립 신호로만 쓰는 방식 검토.)
</div>

<p class="note" style="margin-top:24px">산출: <code>artifacts/cv/cv_manifest.json</code> ·
<code>cv_compare.csv</code> · 각 <code>*/method_summary_model_pred.csv</code>.
Generated by <code>python build_cv_report.py</code> (after <code>groupkfold_cv.py</code>).</p>
</div></body></html>"""
    (CV / "cv_report.html").write_text(html, encoding="utf-8")
    print(f"saved -> {CV/'cv_report.html'}")


if __name__ == "__main__":
    main()
