"""이 서버에서만 계산할 수 있는 것을 파일로 내보낸다 → 운영 PC(Windows)는 exports/ 만 읽는다.

  /home/kptae/miniconda3/envs/qnt_algorithm/bin/python agent_system/web_agent/export_snapshots.py
산출: exports/cells25.parquet (night_axes25 전 모델의 셀 표, _cells25 와 같은 경로) · exports/queue_status.txt · exports/manifest.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exports"
QNT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "tools"))
import _cells25 as c25  # noqa: E402  (import 시 cwd 가 2.5 저장소로 바뀜)
import pandas as pd  # noqa: E402


def main():
    OUT.mkdir(exist_ok=True)
    ax = pd.read_csv(f"{c25.ROOT}/newmodel/data/night_axes25.csv")
    frames, bad = [], []
    for canon in ax.canon:
        r = c25.cells(canon)
        if "error" in r:
            bad.append(canon)
            continue
        t = pd.DataFrame(r["table"])
        t.insert(0, "canon", canon)
        t["source"] = r["source"]
        frames.append(t)
    cells = pd.concat(frames, ignore_index=True)
    tmp = OUT / "cells25.parquet.tmp"
    cells.to_parquet(tmp, index=False)
    tmp.replace(OUT / "cells25.parquet")      # 읽는 쪽이 반쯤 쓴 파일을 보지 않게 교체
    q = subprocess.run(["bash", str(QNT_ROOT / "scripts/gpu_queue_status.sh")], capture_output=True, text=True, timeout=120)
    (OUT / "queue_status.txt").write_text(q.stdout)
    man = {"exported_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cells25_models": int(cells.canon.nunique()),
           "cells25_rows": len(cells), "cells25_missing": bad}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(man, ensure_ascii=False))


if __name__ == "__main__":
    main()
