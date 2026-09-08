#!/usr/bin/env python3
"""셀 이미지 캐시를 미리 만들어 두는 스크립트.

라벨링 중 첫 로드가 느린 것은 safetensors -> PNG 추출(샘플당 약 0.3초, 머신이 바쁘면 10초 이상)
때문이다. 라벨링할 범위를 미리 지정해 캐시를 채워두면 클릭 즉시 뜬다.
서버를 켜 둔 채로 돌려도 된다 (샘플별 락으로 중복 추출은 일어나지 않는다).

예시:
    # FDA_Clinical 의 MP 약제 전체를 4개 프로세스로 미리 추출
    python prefetch.py --source fda_clinical --drug MP --workers 4

    # K. pneumoniae 만, 아직 라벨 안 한 것만
    python prefetch.py --microbial "K. pneumoniae" --todo-only

    # 얼마나 걸릴지/몇 개인지만 확인
    python prefetch.py --drug MP --dry-run
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app  # noqa: E402


def _extract_one(idx: int) -> tuple:
    """워커 프로세스에서 샘플 하나를 추출한다."""
    s = app.SAMPLES_BY_IDX[idx]
    t = time.time()
    try:
        app.ensure_cells(s)
        return idx, True, time.time() - t, ""
    except Exception as e:  # noqa: BLE001
        return idx, False, time.time() - t, str(e)


def select_samples(args) -> list:
    out = []
    for s in app.SAMPLES:
        if args.source and s["src_id"] != args.source:
            continue
        if args.drug and s["antimicrobial"] not in args.drug:
            continue
        if args.organism and s["organism_group"] not in args.organism:
            continue
        if args.microbial and s["microbial_id"] not in args.microbial:
            continue
        if args.todo_only:
            lab = app.LABELS.get(app.label_key(s))
            if lab and lab.get("choice"):
                continue
        out.append(s)
    return out


def main():
    p = argparse.ArgumentParser(description="셀 이미지 캐시 미리 만들기")
    p.add_argument("--source", choices=[s["id"] for s in app.SOURCES], help="데이터 소스")
    p.add_argument("--drug", nargs="+", help="약제 (여러 개 가능, 예: MP PTZ)")
    p.add_argument("--organism", nargs="+", help="organism_group")
    p.add_argument("--microbial", nargs="+", help="microbial_id")
    p.add_argument("--todo-only", action="store_true", help="아직 라벨 안 한 샘플만")
    p.add_argument("--limit", type=int, help="최대 개수")
    p.add_argument("--workers", type=int, default=4, help="병렬 프로세스 수 (기본 4)")
    p.add_argument("--dry-run", action="store_true", help="대상만 세어보고 종료")
    args = p.parse_args()

    sel = select_samples(args)
    todo = [s for s in sel if not app.is_cached(s)]
    if args.limit:
        todo = todo[: args.limit]

    print(f"대상 {len(sel)}개 · 이미 캐시됨 {len(sel) - len([s for s in sel if not app.is_cached(s)])}개")
    print(f"추출할 샘플: {len(todo)}개  (예상 용량 약 {len(todo) * 2 / 1024:.1f} GB)")
    if args.dry_run or not todo:
        return

    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_extract_one, s["idx"]): s for s in todo}
        for fut in as_completed(futs):
            idx, ok, dt, err = fut.result()
            done += 1
            if not ok:
                fail += 1
                s = app.SAMPLES_BY_IDX[idx]
                print(f"  실패 {s['sample_id']} {s['antimicrobial']}: {err}")
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el
                eta = (len(todo) - done) / rate if rate else 0
                print(
                    f"  {done}/{len(todo)}  {rate:.1f}개/초  경과 {el/60:.1f}분  "
                    f"남은 예상 {eta/60:.1f}분",
                    flush=True,
                )

    print(f"완료: {done - fail}개 성공, {fail}개 실패, {(time.time()-t0)/60:.1f}분 소요")


if __name__ == "__main__":
    main()
