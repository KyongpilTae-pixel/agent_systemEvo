# genus 단위 isotonic 보정 — 구현·학습·운영 설명서

## 1. 무엇 / 왜

**배포 환경은 species(organism_group)가 아니라 genus 만 알 수 있다.** dRAST 운영 시점에 들어오는
정보는 genus 수준이므로, per-cell isotonic 보정도 `(organism_group, antimicrobial)` 가 아니라
**`(genus, antimicrobial)`** 단위로 학습·적용해야 배포 현실과 일치한다.

평가(FDA PASS)는 임상 규약대로 **organism_group × antimicrobial** 단위로 집계한다(verify 고정).
즉 **학습·적용은 genus, summary는 organism_group**.

### held-out 근거 (정직한 일반화)

| cell_by | in-sample PASS | **held-out OOF PASS** | 학습 곡선 |
|---|---|---|---|
| organism_group | 103 | 87 | 278 |
| **genus** | 102 (−1) | **91 (+4)** | 170 (genus 14종) |

- **in-sample**: organism 이 약간 우위(세밀 cell 이 자기 데이터에 더 fit = overfit 경향).
- **held-out OOF**(GroupKFold sample_id): **genus 가 +4 우위**. genus cell 은 표본이 더 커
  (예 Staphylococcus n=5370 vs S.aureus 1440) isotonic 이 덜 overfit → 일반화↑.
- 즉 배포 제약(genus 만 앎)을 지키면서 **정직한 일반화는 오히려 더 좋다**.

> ⚠ **운영 채택은 보류**. 위 +4 는 threshold 0.65 단일 지점 기준. routed_iso_t65 를 genus 로
> 교체하려면 **전체 threshold × VME budget sweep 재검증**이 필요(§6). 현재 운영 권고는 organism-iso
> `routed_iso_t65`(in-sample PASS 103 / VME 1.99%) 유지.

## 2. 구현

`agent_system/methods/calibrate/isotonic.py` — `PerCellIsotonic(cell_by="genus")`.

```python
PerCellIsotonic(mode="insample", cell_by="genus")   # 전체 적합 + 보정
PerCellIsotonic(mode="oof",      cell_by="genus")    # GroupKFold(sample_id) held-out
PerCellIsotonic(mode="lookup",   cell_by="genus", lookup_csv=...)  # 동결 자산 로드
```

내부 동작(`_prep`): 기존 검증 클래스 `claudeCode/isotonic_lookup.PerCellIsotonicLookup` 은
`organism_group` 컬럼으로 그룹하므로, genus 모드는 **작업 프레임의 organism_group 자리에 genus
값을 매핑**해 그룹 단위만 genus 로 바꾼다. 실제 organism_group 은 출력 프레임이 보존하므로 verify
summary 는 organism_group 으로 집계된다. (ng_score↔dtw_model_pred 경계 번역도 동시 수행.)

입력은 통합 단일 파일 `data/unified_per_conc.parquet` 의 `genus` 컬럼(= dtw_per_conc.csv 의 genus,
100% 존재; organism_group 첫 단어와도 일치).

## 3. 학습 (배포 자산 동결)

```bash
python -m agent_system.methods.save_genus_iso_lookup
# → agent_system/output/isotonic_lookup_genus.csv  (170 곡선, genus 14종)
```

- 학습 입력 = **routed ng_score(=routed model_pred) ↔ gt_gng**, genus 단위.
- 조건 미달 cell(`min_rows<30` 또는 클래스당 `<3`)은 학습 제외 → identity fallback(raw 유지).
- **저장 CSV 규약**: `organism_group` 컬럼에 **genus 값**을 담는다(키 = (genus, antimicrobial)).
  런타임은 반드시 `cell_by="genus"` 로 로드해야 행의 genus 가 그 키에 매칭된다.

코드에서 직접 학습만:
```python
from agent_system.methods.calibrate.isotonic import PerCellIsotonic
iso = PerCellIsotonic(cell_by="genus").fit_lookup(routed,
        save_path="agent_system/output/isotonic_lookup_genus.csv")
```

## 4. 운영(런타임) 보정

```python
from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

rec = Recipe(
    score     = PerCellModelRouting(),                    # cell-routed model_pred
    calibrate = PerCellIsotonic(mode="lookup", cell_by="genus",
                                lookup_csv="agent_system/output/isotonic_lookup_genus.csv"),
    threshold = 0.65)
rec.evaluate(load_unified())     # → {'pass': 102, ...}  (배포 자산 = in-sample full-fit)
```

런타임은 sklearn 불필요(저장된 x/y_thresholds 에 `np.interp`). 미커버 genus-cell 은 identity.

검증: 위 lookup 경로가 in-sample genus apply 와 **PASS 102 정확 일치**(allclose).

## 5. 파일

| 파일 | 역할 |
|---|---|
| `calibrate/isotonic.py` | `PerCellIsotonic(cell_by="genus")` 구현(+`_prep`, `fit_lookup`) |
| `save_genus_iso_lookup.py` | genus 자산 학습·동결 → `isotonic_lookup_genus.csv` |
| `genus_vs_organism_iso.py` | organism vs genus 비교(in-sample/OOF, summary=organism_group) |
| `agent_system/output/isotonic_lookup_genus.csv` | 동결 배포 자산(170 곡선, organism_group 컬럼=genus) |

## 6. 운영 채택 전 재검증 (Next)

genus-iso 로 `routed_iso_t65` 를 갱신하려면:
1. **threshold × VME budget sweep**: genus-iso 의 t∈{0.50,0.65,0.75} held-out PASS/VME 를
   organism-iso 와 같은 격자로 비교(`ThresholdTuner` + OOF). held-out +4 가 VME 상한 1.5% 하에서도
   유지되는지.
2. **VME 안전성**: genus 가 species 차이를 뭉개므로 특정 species(예 LNZ gram-positive)에서 VME 가
   늘지 않는지 cell-level 확인.
3. 통과 시 운영 자산을 `isotonic_lookup_genus.csv` 로 교체, README/OPERATIONAL_DEPLOYMENT 갱신.

근거 리포트: `genus_vs_organism_iso.py` 출력 + (held-out FAIL→PASS 사례) `build_flip_report.py`.
