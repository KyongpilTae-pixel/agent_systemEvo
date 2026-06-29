"""Multi-Agent FDA PASS 최적화 — Cycle 0~3 종합 비교 보고서.

ISO calibration + threshold tuning의 trade-off를 한 페이지에서 시각화.
LNZ cells 추적 + 운영 권고 옵션 매트릭스.

Output: agent_system/output/cycles_summary_report.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# All cycles to compare
CYCLES = [
    # (label, summary_csv_path, threshold, recipe_desc)
    ("baseline raw (Cycle 0)",
     "claudeCode/output_dtw_aggregate_full/method_summary_model_pred.csv",
     0.5, "same_mic.pt 단독, ISO 없음"),
    ("routed raw",
     "claudeCode/output_sir_reeval_routed/method_summary_model_pred.csv",
     0.5, "PerCellModelRouter, ISO 없음"),
    ("C1 shifted_rankavg3 (no ISO)",
     "claudeCode/output_sir_eval_shifted_rankavg3/method_summary_model_pred.csv",
     0.5, "shifted×rank_avg(beta_lactam+gram_negative+GBM), ISO 없음 — calibration mismatch"),
    ("C2 shifted_rankavg3 + ISO",
     "claudeCode/output_sir_eval_shifted_rankavg3_iso/method_summary_model_pred.csv",
     0.5, "rank_avg을 ISO LOSO 재학습"),
    ("C2b baseline + ISO",
     "claudeCode/output_sir_eval_baseline_iso/method_summary_model_pred.csv",
     0.5, "baseline model_pred에 ISO LOSO 재학습"),
    ("C2c routed + ISO",
     "claudeCode/output_sir_eval_routed_iso/method_summary_model_pred.csv",
     0.5, "routed model_pred에 ISO LOSO 재학습 — PASS 최대"),
    ("C3a baseline + ISO (thr 0.75)",
     "claudeCode/output_sir_eval_baseline_iso_t75/method_summary_model_pred.csv",
     0.75, "C2b + threshold 0.75 (운영 권고)"),
    ("C3b routed + ISO (thr 0.75)",
     "claudeCode/output_sir_eval_routed_iso_t75/method_summary_model_pred.csv",
     0.75, "C2c + threshold 0.75 (운영 권고)"),
    ("C3c routed + ISO (thr 0.65)",
     "claudeCode/output_sir_eval_routed_iso_t65/method_summary_model_pred.csv",
     0.65, "C2c + threshold 0.65 (sweet spot)"),
]
LNZ_CELLS = [
    "Staphylococcus Coagulase-negative", "Staphylococcus aureus",
    "Staphylococcus lugdunensis", "Enterococcus faecalis",
    "Enterococcus faecium", "Enterococcus gallinarum",
    "Enterococcus avium", "Enterococcus casseliflavus",
]


def _pass_count_lnz(csv_path: Path) -> tuple[int, int, int, int]:
    """Return (pass, total, lnz_pass, lnz_total)."""
    if not csv_path.exists():
        return (0, 0, 0, 0)
    df = pd.read_csv(csv_path)
    df["_pass"] = df["FDA_fail_list_exception_rule"].astype(str).str.strip() == "PASS"
    lnz = df[df["antimicrobial"] == "LNZ"]
    return (int(df["_pass"].sum()), int(len(df)),
            int(lnz["_pass"].sum()), int(len(lnz)))


def _feasibility_metrics(csv_path: Path) -> dict:
    """model_pred row의 ea/ca/me/vme/auroc."""
    feas = csv_path.parent / "method_sir_feasibility.csv"
    if not feas.exists():
        return {}
    f = pd.read_csv(feas)
    r = f[f.method == "model_pred"]
    if len(r) == 0:
        return {}
    r = r.iloc[0]
    return {"auroc": float(r["auroc"]), "ea": float(r["ea_rate"]),
            "ca": float(r["ca_rate"]), "me": float(r["me_rate"]),
            "vme": float(r["vme_rate"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",
                   default=Path("agent_system/output/cycles_summary_report.html"),
                   type=Path)
    args = p.parse_args()

    rows = []
    for label, path, thr, desc in CYCLES:
        path = Path(path)
        np_pass, n_total, lnz_pass, lnz_total = _pass_count_lnz(path)
        metrics = _feasibility_metrics(path)
        rows.append({"label": label, "thr": thr, "desc": desc,
                     "pass": np_pass, "total": n_total,
                     "lnz_pass": lnz_pass, "lnz_total": lnz_total,
                     **metrics})

    # baseline reference
    baseline_pass = rows[0]["pass"]
    summary_rows = ""
    for i, r in enumerate(rows):
        bg = ""
        if "C2c" in r["label"]:
            bg = "background:#fff3cd;"
        elif "C3c" in r["label"]:
            bg = "background:#d4edda;"
        elif "C3b" in r["label"]:
            bg = "background:#d1ecf1;"
        delta = r["pass"] - baseline_pass
        delta_str = f'<b style="color:{"#2e7d32" if delta>0 else "#c62828"}">{delta:+d}</b>'
        vme_alert = ""
        if r.get("vme"):
            vme_alert = " ⚠" if r["vme"] > 0.025 else ""
        summary_rows += (
            f"<tr style='{bg}'><td>{r['label']}</td><td>{r['thr']}</td>"
            f"<td><b>{r['pass']}</b> / {r['total']}</td><td>{delta_str}</td>"
            f"<td>{r['lnz_pass']} / {r['lnz_total']}</td>"
            f"<td>{r.get('ea', 0)*100:.1f}%</td>"
            f"<td>{r.get('ca', 0)*100:.1f}%</td>"
            f"<td>{r.get('me', 0)*100:.1f}%</td>"
            f"<td>{r.get('vme', 0)*100:.2f}%{vme_alert}</td>"
            f"<td style='font-size:11px;color:#555;'>{r['desc']}</td></tr>")

    # Best PASS, best VME-tolerant
    best_pass = max(rows, key=lambda r: r["pass"])
    safe_pass = max([r for r in rows if r.get("vme", 0) <= 0.020], key=lambda r: r["pass"],
                    default=best_pass)
    strict_pass = max([r for r in rows if r.get("vme", 0) <= 0.015], key=lambda r: r["pass"],
                      default=safe_pass)

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>Cycles 0~3 종합 비교 — FDA PASS 최적화</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,sans-serif; margin:0;
          background:#fafafa; color:#222; line-height:1.55; }}
  .c {{ max-width:1280px; margin:0 auto; padding:24px 32px 80px; }}
  h1 {{ font-size:23px; border-bottom:2px solid #2979ff; padding-bottom:6px; }}
  h2 {{ font-size:17px; color:#1565c0; margin-top:30px; }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
  th,td {{ border:1px solid #e0e0e0; padding:6px 9px; text-align:right;
           vertical-align:top; }}
  th {{ background:#f5f7fa; font-size:11.5px; }}
  td:first-child,th:first-child {{ text-align:left; }}
  td:last-child,th:last-child {{ text-align:left; }}
  code {{ background:#f0f0f0; padding:1px 5px; border-radius:3px; font-size:11.5px; }}
  blockquote {{ border-left:4px solid #2979ff; padding:6px 14px; background:#f5f9ff;
                color:#444; margin:10px 0; font-size:12.5px; }}
  .card {{ background:#fff; border:1px solid #ddd; border-radius:8px;
           padding:14px 18px; margin:14px 0; }}
  .insight {{ background:#e8f5e9; border-left:4px solid #2e7d32;
              padding:10px 16px; margin:14px 0; font-size:13px; }}
  .warn {{ background:#fff3e0; border-left:4px solid #ef6c00;
           padding:10px 16px; margin:14px 0; font-size:13px; }}
</style></head><body><div class="c">
<h1>Multi-Agent FDA PASS 최적화 — Cycles 0~3 종합 비교</h1>
<p>2026-05-27 진행한 cycle 0 (baseline 재현) → cycle 3 (ISO + threshold sweep)
까지 9가지 운영 옵션의 PASS / EA / CA / ME / VME trade-off.</p>

<blockquote>
<b>핵심 발견 1</b>: ISO calibration LOSO 재학습이 단독으로 PASS +30 (49→79). routing은
+2 추가 (79→81). rank_avg ensemble은 ISO 통한 후 0의 추가 효과.<br>
<b>핵심 발견 2</b>: threshold sweep으로 VME 통제 가능 (thr 0.5→0.75 VME 3.45%→1.50%).
PASS 81→61. trade-off 단순.<br>
<b>운영 권고</b>: 아래 3 옵션 중 선택 (VME 허용도에 따라).
</blockquote>

<h2>📊 전체 cycles 비교 (9 옵션)</h2>
<div class="card"><table>
<thead><tr><th>recipe</th><th>thr</th><th>FDA PASS</th><th>Δ vs baseline</th>
<th>LNZ PASS</th><th>EA</th><th>CA</th><th>ME</th><th>VME</th>
<th>설명</th></tr></thead>
<tbody>{summary_rows}</tbody></table>
<p style="font-size:11.5px;color:#666;">색상: 🟡 = PASS 최대 (VME 위험), 🟢 =
sweet spot (PASS 높고 VME 통제), 🔵 = 운영 권고 (VME ≤ 1.5%). ⚠ = VME > 2.5%.</p>
</div>

<h2>🎯 PASS vs VME Pareto 분석</h2>
<div class="insight">
<b>PASS 최대 (VME 무시)</b>: <code>{best_pass['label']}</code> →
<b>{best_pass['pass']} PASS</b>, VME {best_pass.get('vme', 0)*100:.2f}%<br>
<b>VME ≤ 2% (sweet spot)</b>: <code>{safe_pass['label']}</code> →
<b>{safe_pass['pass']} PASS</b>, VME {safe_pass.get('vme', 0)*100:.2f}%<br>
<b>VME ≤ 1.5% (FDA 엄격)</b>: <code>{strict_pass['label']}</code> →
<b>{strict_pass['pass']} PASS</b>, VME {strict_pass.get('vme', 0)*100:.2f}%
</div>

<h2>🔍 Key Insights</h2>

<div class="card">
<h3 style="color:#1565c0;margin:0 0 8px;">1. ISO calibration이 최대 lever</h3>
<p>baseline raw 49 → baseline + ISO 79 (+30 PASS). routing 추가는 +2만.
rank_avg는 ISO 후 0 추가. 이전 추정(routing > ISO)이 뒤집힘.</p>
</div>

<div class="card">
<h3 style="color:#1565c0;margin:0 0 8px;">2. Threshold가 VME 직접 통제</h3>
<p>thr 0.5 → 0.75로 변경 시 VME 3.45% → 1.50% (R-conservative). PASS는
81 → 61 떨어짐. <b>threshold 0.65가 sweet spot</b> — PASS 71, VME 1.99%.</p>
</div>

<div class="card">
<h3 style="color:#1565c0;margin:0 0 8px;">3. C1 (rank_avg without ISO) = 실패</h3>
<p>shifted×rank_avg ensemble만 적용하면 PASS 44 (baseline 49보다도 −5).
원인: rank_avg는 probability 의미가 없는데 threshold 0.5가 직접 적용되어
calibration mismatch. RANK_AVERAGE_ENSEMBLE.md Section 6 CRITICAL warning 실증.</p>
</div>

<div class="card">
<h3 style="color:#1565c0;margin:0 0 8px;">4. ISO LOSO refit은 in-domain leakage 있음</h3>
<p>ISO 곡선은 같은 cell의 다른 sample을 사용해 fit. cell 단위 leakage 없음
(GroupKFold by sample_id). 그러나 cell 자체는 평가 시 알려진 cell이므로 in-FDA
domain 최적. Train→FDA transfer는 별도 검증 필요 (Phase 3 Overfit Sentinel 작업).</p>
</div>

<h2>🛠 운영 적용 옵션 (사용자 선택)</h2>
<div class="card"><table>
<thead><tr><th>option</th><th>recipe</th><th>PASS</th><th>VME</th><th>적용 권고</th></tr></thead>
<tbody>
<tr><td><b>적극</b></td><td><code>routed + ISO + thr 0.50</code></td>
<td><b>81</b></td><td>3.45% ⚠</td>
<td>PASS 우선, VME 임상 안전성 양보. 운영 deploy 전 medical review 필수</td></tr>
<tr style="background:#d4edda;"><td><b>균형 ★</b></td>
<td><code>routed + ISO + thr 0.65</code></td>
<td><b>71</b></td><td>1.99%</td>
<td><b>운영 권고</b>: PASS +13 (vs baseline routed 58), VME 2% 미만. FDA 룰
exception 적용으로 cell 단위 통과 71</td></tr>
<tr><td><b>보수</b></td><td><code>routed + ISO + thr 0.75</code></td>
<td><b>61</b></td><td>1.50% ✓</td>
<td>VME FDA 한계 정확히 통제 (1.5%). PASS +3 (vs baseline routed)</td></tr>
</tbody></table>
</div>

<h2>📂 산출 디렉토리</h2>
<div class="card"><ul style="font-size:12px;">
<li><code>claudeCode/output_sir_eval_baseline_iso/</code> — C2b</li>
<li><code>claudeCode/output_sir_eval_routed_iso/</code> — C2c (PASS 81)</li>
<li><code>claudeCode/output_sir_eval_shifted_rankavg3_iso/</code> — C2</li>
<li><code>claudeCode/output_sir_eval_baseline_iso_t75/</code> — C3a</li>
<li><code>claudeCode/output_sir_eval_routed_iso_t75/</code> — C3b</li>
<li><code>claudeCode/output_sir_eval_routed_iso_t65/</code> — C3c (sweet spot)</li>
</ul></div>

<h2>📑 관련 자산</h2>
<div class="card"><ul style="font-size:12px;">
<li><code>agent_system/refit_iso_on_per_conc.py</code> — ISO LOSO refit helper</li>
<li><code>agent_system/build_per_conc_for_recipe.py</code> — recipe → per_conc CSV</li>
<li><code>claudeCode/verify_method_sir_pipeline.py</code> — <code>--native_threshold</code> CLI arg 추가됨</li>
<li><code>agent_system/agents/sir_evaluator.py</code> — D.evaluate subprocess wiring</li>
<li><code>agent_system/agents/overfit_sentinel.py</code> — E.validate_cv GroupKFold(5)</li>
<li><code>agent_system/sdk_tools.json</code> — 8 agents × 23 tools SDK schema</li>
</ul></div>

<p style="color:#888;font-size:12px;margin-top:30px;">Generated 2026-05-28.
<code>python -m agent_system.build_cycles_summary_report</code>.</p>
</div></body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
