"""2.5 셀 단위 임상 지표 — night_axes25 와 **같은 경로**(파일 선택·load·K4 중복제거·cell_table)를 재사용.

qnt_algorithm conda env(lightgbm 필요)에서 subprocess 로 실행된다:
  python _cells25.py cells <canon>        → JSON (셀 표)
  python _cells25.py verify [N]           → night_axes25.csv 의 clin_pass/clin_cells 와 대조
자체 구현 금지 규약: 계산은 전부 newmodel.build_night_axes25 / build_pass_report 함수.
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout

ROOT = "/home/kptae/project/drast_25_lrcn"
with redirect_stdout(io.StringIO()):          # 모듈 import 시 출력 억제 (JSON stdout 보호)
    sys.path.insert(0, ROOT)
    import newmodel.build_night_axes25 as na  # noqa: E402  (import 시 cwd=ROOT 로 바뀜)
    from newmodel.build_pass_report import cell_table  # noqa: E402


def clin_path(c: str) -> str | None:
    """night_axes25.main 과 같은 우선순위: judge25/base_clin → 큐 추론 체인 --out → pred_auto."""
    nb = f"{na.OUT}/judge25/base_clin_{c}.parquet"
    if os.path.exists(nb):
        return nb
    # ★trainall_* 는 학습 패널까지 포함한 추론(--split_all) — 임상 평가에 쓰면 학습 데이터 오염(2026-09-30 검증에서 발견)
    for f in na.queue_rows():
        cc, paths = na.paths_of(f[7])
        cp = paths.get("clin") or ""
        if cc == c and cp and "trainall" not in os.path.basename(cp) and os.path.exists(cp):
            return cp
    p = na.default_paths(c)["clin"]
    return p if os.path.exists(p) else None


def cells(c: str):
    p = clin_path(c)
    if p is None:
        return {"error": f"임상 예측 파일 없음: {c}"}
    d = na.load(p, c).drop_duplicates(na.K4)
    ct = cell_table(d, "x")
    ev = ct[ct.passed.notna()]
    return {"canon": c, "source": os.path.join(ROOT, p) if not p.startswith("/") else p,
            "cells": len(ev), "pass": int((ev.passed == True).sum()),  # noqa: E712
            "table": json.loads(ev.drop(columns=["model"], errors="ignore").to_json(orient="records"))}


def verify(n: int):
    import pandas as pd
    ax = pd.read_csv(f"{ROOT}/newmodel/data/night_axes25.csv").dropna(subset=["clin_pass"])
    bad = 0
    for _, r in ax.head(n).iterrows():
        out = cells(r.canon)
        ok = out.get("pass") == int(r.clin_pass) and out.get("cells") == int(r.clin_cells)
        bad += not ok
        print(("OK  " if ok else "DIFF"), r.canon[:70], out.get("pass"), out.get("cells"), "| axes", int(r.clin_pass), int(r.clin_cells))
    print("mismatch", bad, "/", min(n, len(ax)))


if __name__ == "__main__":
    if sys.argv[1] == "cells":
        print(json.dumps(cells(sys.argv[2]), ensure_ascii=False, default=str))
    else:
        verify(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
