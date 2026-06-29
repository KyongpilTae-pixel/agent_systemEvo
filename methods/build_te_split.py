"""Technical error(TE) split — clean 평가셋과 TE 별도 관리 자산 생성.

기준: claudeCode/data/FDA_Clinical_selected_model.xlsx 의 all_concentration sheet 의 sample_id
= QuantaMatrix 가 TE 를 제거한 clean set. 현재 통합 입력(unified_per_conc.parquet)에서 이 목록에
없는 sample 은 Technical error 로 분리한다.

산출:
  data/unified_per_conc_clean.parquet  — TE 제거 평가셋
  data/unified_per_conc_te.parquet     — 제외된 TE sample (별도 관리)
  data/te_samples.csv                  — TE sample_id 목록(+organism_group/genus)
  data/te_split_manifest.json          — 카운트/출처

사용: python -m agent_system.methods.build_te_split
"""
from __future__ import annotations

import json

import pandas as pd

from agent_system.methods import base as B

CLEAN_SHEET = "all_concentration"


def clean_sample_ids() -> set:
    df = pd.read_excel(B.CLEAN_SOURCE_XLSX, sheet_name=CLEAN_SHEET, usecols=["sample_id"])
    return set(df["sample_id"].astype(str).unique())


def main() -> None:
    if not B.UNIFIED_INPUT.exists():
        raise FileNotFoundError(f"{B.UNIFIED_INPUT} 없음 — build_unified_input 먼저")
    u = pd.read_parquet(B.UNIFIED_INPUT)
    clean_ids = clean_sample_ids()
    sid = u["sample_id"].astype(str)
    is_clean = sid.isin(clean_ids)

    clean = u[is_clean].reset_index(drop=True)
    te = u[~is_clean].reset_index(drop=True)
    clean.to_parquet(B.UNIFIED_CLEAN, index=False)
    te.to_parquet(B.UNIFIED_TE, index=False)

    # TE sample 목록 (관리용) — sample 단위 메타
    te_meta = (te[["sample_id", B.OG_COL, B.GENUS_COL]].drop_duplicates("sample_id")
               .sort_values("sample_id").reset_index(drop=True))
    te_meta["reason"] = "technical_error"
    te_meta["source"] = "FDA_Clinical_selected_model.xlsx:all_concentration (not listed)"
    te_meta.to_csv(B.TE_SAMPLES_CSV, index=False)

    manifest = {
        "source_xlsx": str(B.CLEAN_SOURCE_XLSX), "clean_sheet": CLEAN_SHEET,
        "clean_unique_sample_ids": len(clean_ids),
        "fda_unique_sample_ids": int(sid.nunique()),
        "kept_samples": int(clean["sample_id"].nunique()),
        "te_samples": int(te["sample_id"].nunique()),
        "kept_rows": int(len(clean)), "te_rows": int(len(te)), "total_rows": int(len(u)),
    }
    (B.DATA_DIR / "te_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"[clean] {B.UNIFIED_CLEAN}")
    print(f"  sample {manifest['kept_samples']} / rows {manifest['kept_rows']:,}")
    print(f"[TE]    {B.UNIFIED_TE}  (별도 관리)")
    print(f"  sample {manifest['te_samples']} / rows {manifest['te_rows']:,}")
    print(f"[목록]  {B.TE_SAMPLES_CSV}")
    # TE sample 의 organism_group 분포(상위)
    vc = te_meta[B.OG_COL].value_counts()
    print(f"\nTE sample organism_group 분포(상위):")
    for og, n in vc.head(12).items():
        print(f"  {og:36s} {n}")


if __name__ == "__main__":
    main()
