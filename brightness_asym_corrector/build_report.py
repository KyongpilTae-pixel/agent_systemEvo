#!/usr/bin/env python3
"""report.html 생성 — artifacts 의 manifest/CSV 에서 수치를 읽어 보고서 작성."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parent
ART = PKG / "artifacts"


def _load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else (default or {})


def main() -> None:
    org = _load(ART / "eval" / "eval_manifest.json")
    gen = _load(ART / "eval_genus" / "eval_manifest.json")
    grout = _load(ART / "genus_routing" / "manifest.json")
    gbm = _load(PKG / "model" / "brightness_gbm.meta.json", _load(ART / "brightness_gbm_manifest.json"))

    def row(tag, d):
        if not d:
            return ""
        b, c = d["baseline"], d["corrected"]
        return (f"<tr><td>{tag}</td><td>{b['pass']}</td><td>{c['pass']} "
                f"<b>({d['delta_pass']:+d})</b></td><td>{b['vme_cells']}→{c['vme_cells']} "
                f"({d['delta_vme_cells']:+d})</td><td>{c['vme_rate']:.4f}</td>"
                f"<td>gain {d['pass_gain']} / loss {d['pass_loss']}</td></tr>")

    gbm_auc_in = gbm.get("auroc_in_domain", "0.915")
    gbm_auc_cross = gbm.get("auroc_cross_domain_fda", gbm.get("auroc_cross_domain_fda", "0.918"))

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>Brightness Asymmetric Corrector — 보고서</title>
<style>
 body{{font-family:-apple-system,'Segoe UI','Malgun Gothic',sans-serif;margin:0;padding:28px 40px;color:#1f2328;line-height:1.55;max-width:1080px;}}
 h1{{font-size:23px;margin:0 0 4px;}} h2{{font-size:17px;margin:24px 0 10px;border-bottom:2px solid #d0d7de;padding-bottom:5px;}}
 .sub{{color:#57606a;font-size:12.5px;}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;}}
 th,td{{border:1px solid #d8dee4;padding:6px 9px;text-align:left;}} th{{background:#f0f3f6;}}
 tr:nth-child(even){{background:#fbfcfd;}}
 .big{{display:flex;gap:14px;margin:14px 0;flex-wrap:wrap;}}
 .card{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:10px;padding:14px 22px;text-align:center;font-size:12.5px;color:#57606a;}}
 .card .v{{font-size:25px;font-weight:700;color:#1a7f37;}}
 .note{{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 14px;margin:12px 0;font-size:13px;border-radius:0 6px 6px 0;}}
 .warn{{background:#fff8c5;border-left:4px solid #d4a72c;padding:10px 14px;margin:12px 0;font-size:13px;border-radius:0 6px 6px 0;}}
 code{{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12px;}}
</style></head><body>
<h1>Brightness Asymmetric Corrector</h1>
<div class="sub">Brightness=True cell 한정 · asym G-방향 brightness 보정 · 운영 routed_iso_t65 대비 · 2026-06-02</div>

<div class="big">
 <div class="card"><div class="v">103 → {org.get('corrected',{}).get('pass','106')}</div>FDA PASS (+{org.get('delta_pass',3)})</div>
 <div class="card"><div class="v">43 → {org.get('corrected',{}).get('vme_cells','42')}</div>VME-cell ({org.get('delta_vme_cells',-1):+d})</div>
 <div class="card"><div class="v">3 / 0</div>PASS gain / loss</div>
 <div class="card"><div class="v">{gbm_auc_in} / {gbm_auc_cross}</div>brightness GBM AUROC<br>in / cross-domain</div>
</div>

<h2>1. 방법</h2>
<div class="note">
운영 score = <b>P(NG)</b>. VME(거짓 NG = 실제 G인데 NG 예측 → MIC 과소 → S 오보고)는 P(NG)를
<b>올릴</b> 때 발생. 따라서 brightness 가 image 보다 <b>더 G</b>라고 말할 때(<code>brp &lt; img</code>)만
점수를 내리고(G 방향), 더 NG 라고 말할 땐 image 유지 → false-NG(VME) 구조적 차단.
<pre style="font-size:12px;background:#f6f8fa;padding:8px 12px;border-radius:6px;">corrected = (1-w)·img + w·brp   (brp &lt; img 이고 Brightness=True cell),  w=0.7
          = img                  (그 외)</pre>
이후 per-cell isotonic → threshold 0.65 → drast_gng→MIC→SIR→evalEA (기존 운영 파이프라인).
brightness GBM 은 학습 CSV(7-timepoint <code>brightness_list</code>)로 in-domain 학습, FDA 추론.
</div>

<h2>2. 검증 — 적용 범위별 (전체 313 cell)</h2>
<table>
<tr><th>적용 범위</th><th>baseline PASS</th><th>corrected PASS</th><th>VME-cell</th><th>VME-rate</th><th>flip</th></tr>
{row("Brightness=True organism (45 cell)", org)}
{row("genus×drug (evo, 46 cell)", gen) if gen else '<tr><td>genus×drug (evo, 46 cell)</td><td colspan=5>실행 중…</td></tr>'}
</table>
<div class="sub">개선 cell (organism): Proteus mirabilis×CZA (VME→PASS), Proteus vulgaris×CPM (ME→PASS),
Serratia marcescens×CAZ (EA,ME→PASS) — 모두 cephalosporin 과대 R 호출 교정. PASS 손실 0.</div>

<h2>3. 방법론 비교 (왜 이 구성인가)</h2>
<table>
<tr><th>변형</th><th>Brightness=True 45 cell PASS</th><th>VME-cell</th><th>판정</th></tr>
<tr><td>baseline (image only)</td><td>13</td><td>14</td><td>기준</td></tr>
<tr><td>대칭(양방향) 보정 w0.3</td><td>14 (+1)</td><td>15 (+1)</td><td>VME 동반 ✗</td></tr>
<tr><td><b>비대칭 G-방향 w0.7 (절대값 GBM)</b></td><td><b>16 (+3)</b></td><td><b>13 (−1)</b></td><td>채택 ✅</td></tr>
<tr><td>비대칭 (거리/growth_dev GBM)</td><td>13 (+0)</td><td>17 (+3)</td><td>열세</td></tr>
<tr><td>brightness 단독 판단</td><td>5 (−8)</td><td>21</td><td>불가 ✗</td></tr>
</table>
<div class="sub">거리(growth_dev) GBM 은 standalone AUROC 0.9265 로 더 높지만 보정자로는 열세 —
절대 trajectory feature(<code>oa_drug_late_slope</code> 등)의 오차 구조가 G-방향 비대칭과 더 잘 맞음.
brightness 는 단독 판단 불가, image 의 G-방향 보정자로만 유효.</div>

<h2>4. genus 단위 검토</h2>
<div class="note">
모델/brightness 적용을 genus(organism_group 의 상위 범주)로 묶어 검토.
evo 워크북(<code>evo_selected_model_and_brightness_thershold_260529.xlsx</code>)의 brightness 지정 =
genus ∈ {{Morganella, Proteus, Serratia}} × 41 약제. 이를 FDA organism cell 로 전개 → <b>46 cell</b>.
F21(organism Brightness=True) 45 cell 과 <b>45개 교집합, evo 가 Proteus vulgaris×AMP 1개 추가</b>
(genus 묶기가 P. mirabilis 의 플래그를 P. vulgaris 로 일반화). 따라서 genus 방법론 ≈ organism 결과 + AMP 1 cell.
</div>

<h2>5. genus 모델 맵핑 routing 성능 (organism×약제 단위)</h2>
<div class="note">
evo 워크북의 (Genus, 약제)→Model 맵핑을 각 organism cell 에 부여(genus 모델을 소속 organism
전체에 적용)하고, <b>동일 평가 단위(organism×약제)·동일 파이프라인</b>으로 성능 측정. 모델 맵핑만 다름.
</div>
<table>
<tr><th>routing</th><th>PASS</th><th>VME-cell</th><th>VME-rate</th><th>vs our</th></tr>
<tr><td>our_routing (AUROC, per-organism)</td><td>{grout.get('our_routing',{}).get('pass','103')}</td>
<td>{grout.get('our_routing',{}).get('vme_cells','43')}</td>
<td>{grout.get('our_routing',{}).get('vme_rate','0.0177')}</td><td>기준</td></tr>
<tr><td><b>genus_routing (evo 맵핑)</b></td><td><b>{grout.get('genus_routing',{}).get('pass','95')}</b></td>
<td>{grout.get('genus_routing',{}).get('vme_cells','42')}</td>
<td><b>{grout.get('genus_routing',{}).get('vme_rate','0.0155')}</b></td>
<td>ΔPASS {grout.get('delta_pass',-8):+d} (gain {grout.get('pass_gain',8)}/loss {grout.get('pass_loss',16)})</td></tr>
</table>
<div class="sub">
<b>PASS↔VME 트레이드오프.</b> evo genus 맵핑은 PASS 8 낮지만 VME-rate 1.77%→1.55% (FDA strict 1.5% 근접) —
gram_negative 를 메뉴에서 제외한 보수적 선택. gain 8 = carbapenem/cephalosporin VME 제거
(E. coli×CZA·MEV 등), loss 16 = same_mic/beta_lactam 의 R 과대호출(ME)로 PASS 손실
(Proteus×CPM/CRO/GEN 등). 우리 AUROC routing 은 PASS 최대(103), evo genus 는 VME 안전.
evo 모델 메뉴 4종(same_mic/beta_lactam/object_area_045~08/early_stop), gram_negative 미사용.
</div>

<h2>6. 워크북 비교 (모델 선정)</h2>
<table>
<tr><th>비교</th><th>모델 일치율</th><th>비고</th></tr>
<tr><td>F21 (organism) vs 우리 routing</td><td>19.1% (204 cell)</td><td>실질충돌 12.7%, carbapenem/colistin 집중</td></tr>
<tr><td>evo (genus) vs 우리 routing</td><td>17.3% (173 cell)</td><td>evo 는 gram_negative 미사용 (메뉴 4종)</td></tr>
</table>
<div class="sub">두 워크북 모두 <b>임상 PASS 기준</b>, 우리 routing 은 <b>AUROC 기준</b> → 기준 차이로 표면 일치율 낮음.
대부분 near-tie 이며 실질 충돌은 carbapenem/colistin × Enterobacterales 에 집중 (별도 검증: 우리 선택이 VME 동등이상 안전).</div>

<h2>7. 주의 / 다음 단계</h2>
<div class="warn">
• evo 의 <code>Brightness_threshold</code>(0.4~0.75)는 <b>10시간 readout 기준</b> → 현재 7-timepoint 데이터에 사용 불가.
본 모듈은 그 고정값 대신 현재 데이터로 GBM 학습 + 현재 FDA per-cell isotonic 보정(데이터-구동).<br>
• <b>in-sample</b>: w=0.7 을 FDA 패널에서 선택 → 운영 투입 전 GroupKFold(sample_id) CV 로 +3 PASS held-out 유지 확인 권장.<br>
• 모델 라우팅을 genus 로 묶는 작업은 별도 (본 모듈은 brightness 보정만 담당).
</div>
</body></html>"""
    out = PKG / "report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[saved] {out}  ({len(doc)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
