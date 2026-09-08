# -*- coding: utf-8 -*-
"""★2.5(d170) 라벨링 소스 + 서브셋 생성 (2026-09-08 지시).

왜 — 검토 대상(전 구조 실패 웰 · 운영만 맞히는 패널 · TE 의심)이 전부 **2.5 d170** 인데
  라벨링 도구는 3.0 세 데이터셋만 읽는다. 2.5 를 **네 번째 소스**로 붙인다.

⚠3.0 과 입력 형식이 다르다.
  3.0 : 패널당 `.safetensors` 한 덩어리 `(N,1,224,224)` → 블록을 잘라 셀 PNG
  2.5 : **웰마다 PNG 7장**(`Thumbnail/*.png` 신형 · `*.jpg` 구형) → 경로를 그대로 넘긴다
  그래서 `png_frames`(행×시점 경로 JSON)를 만들어 두고 앱이 복사만 하게 한다.

⚠`bmd_mic_order` 가 d170 에 없다. `bmd_mic` 와 농도 배열로 **계산**한다.
  규약은 앱과 같다 — 0=최저농도 · 14=전 농도 성장(off-scale `>`/`>=`).

★TE(기술오류) 점수를 **격자 행마다** 실어 보낸다 — `newmodel/data/te25_panel_scores.csv`
  (없으면 조용히 건너뛴다). 행 순서는 격자와 같다: control(농도 오름차순) → 약제 농도(오름차순).
  라벨러가 "이 칸이 이상한데 기포인가?" 를 화면에서 바로 확인하게 하려는 것이다.

서브셋 = 사이드바에서 "이것만 보기" 로 거르는 이름표. `subsets/<id>.csv`
  (`project_id,sample_id,antimicrobial`). 패널 단위이며 여러 개가 겹칠 수 있다.

사용: python build_source25.py
출력: /home/kptae/data/allinfo/d25/d170_label_source.parquet · subsets/*.csv
"""
import os
import re
import sys
import json

import numpy as np
import pandas as pd

D25 = '/home/kptae/project/drast_25_lrcn'
sys.path.insert(0, D25)
sys.path.insert(0, '/home/kptae/project/qnt_algorithm')
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = '/home/kptae/data/allinfo/d25'
SUBDIR = os.path.join(HERE, 'subsets')
NFRAME = 7


def mic_order(bmd, concs):
    """bmd_mic + 농도배열 → order(0=최저 · 14=전 농도 성장). 못 정하면 ''."""
    m = re.match(r'^\s*(<=|>=|>|<)?\s*([\d.]+)', str(bmd))
    if not m:
        return ''
    op, v = (m.group(1) or ''), float(m.group(2))
    cs = [float(c) for c in concs]
    if op in ('>', '>='):
        return '14'
    if op in ('<', '<='):
        return '0'
    for i, c in enumerate(cs):
        if abs(c - v) < 1e-9:
            return str(i)
    return '14' if v > max(cs) else ('0' if v < min(cs) else '')


def main():
    os.chdir(D25)
    from newmodel import build_vme_images as V
    from newmodel.wellpath import frames as wp_frames

    d = pd.read_csv(V.D170, dtype=str, low_memory=False)
    d['c'] = pd.to_numeric(d.concentration, errors='coerce')
    print(f'[src25] d170 웰 {len(d):,}', flush=True)

    # ── 검토 대상 패널 모으기 ──
    subs = {}
    p_all = f'{D25}/newmodel/data/recipe_allwrong.csv'
    if os.path.exists(p_all):
        A = pd.read_csv(p_all)
        subs['allwrong'] = (A[['project_id', 'sample_id', 'drug']]
                            .rename(columns={'drug': 'antimicrobial'}).drop_duplicates())
    p_op = f'{D25}/newmodel/data/op_only_correct_panels.csv'
    if os.path.exists(p_op):
        O = pd.read_csv(p_op)
        subs['op_only'] = O[['project_id', 'sample_id', 'antimicrobial']].drop_duplicates()
    p_te = f'{D25}/newmodel/data/te25_scores.csv'
    if os.path.exists(p_te):
        T = pd.read_csv(p_te)
        flag = ((T.get('bubble_h119', 0) >= 0.41) | (T.get('film_h119', 0) >= 0.31))
        subs['te_suspect'] = (T[flag][['project_id', 'sample_id', 'drug']]
                              .rename(columns={'drug': 'antimicrobial'}).drop_duplicates())
    os.makedirs(SUBDIR, exist_ok=True)
    for k, v in subs.items():
        v = v.astype(str)
        v.to_csv(f'{SUBDIR}/{k}.csv', index=False, encoding='utf-8-sig')
        print(f'[src25] subset {k:12s} 패널 {len(v):,}', flush=True)

    want = pd.concat([v.astype(str) for v in subs.values()], ignore_index=True).drop_duplicates()
    want['pk'] = want.project_id + '|' + want.sample_id + '|' + want.antimicrobial
    keep = set(want.pk)
    d['pk'] = d.project_id.astype(str) + '|' + d.sample_id.astype(str) + '|' + d.antimicrobial.astype(str)
    print(f'[src25] 대상 패널 {len(keep):,}', flush=True)

    # ── control 은 검체 단위 ──
    ctl = d[d.antimicrobial == 'Cont'].copy()
    ctl['sk'] = ctl.project_id.astype(str) + '|' + ctl.sample_id.astype(str)
    ctl = ctl.sort_values('c')
    ctl_by = {k: g for k, g in ctl.groupby('sk')}

    # ── TE25 행별 점수(있으면) ──
    te_map = {}
    p_te = f'{D25}/newmodel/data/te25_panel_scores.csv'
    if os.path.exists(p_te):
        T = pd.read_csv(p_te)
        T['k'] = (T.project_id.astype(str) + '|' + T.sample_id.astype(str) + '|'
                  + T.antimicrobial.astype(str) + '|'
                  + pd.to_numeric(T.c, errors='coerce').map(lambda x: f'{x:g}'))
        te_map = {r.k: (round(float(r.bubble), 3), round(float(r.film), 3))
                  for r in T.itertuples()}
        print(f'[src25] TE 점수 {len(te_map):,} 웰', flush=True)
    else:
        print('[src25] ⚠TE 점수 없음 — infer_te25_panels 를 먼저 돌린다', flush=True)

    rows = []
    miss_img = 0
    for pk, g in d[d.pk.isin(keep)].groupby('pk'):
        g = g[g.c.notna()].sort_values('c')
        if not len(g):
            continue
        r0 = g.iloc[0]
        sk = f'{r0.project_id}|{r0.sample_id}'
        cg = ctl_by.get(sk)
        concs = [f'{x:g}' for x in g.c.tolist()]
        grid, te_rows = [], []

        def _te(am, cv):
            return te_map.get(f'{r0.project_id}|{r0.sample_id}|{am}|{cv:g}')

        if cg is not None:
            for _, cr in cg.iterrows():
                grid.append(wp_frames(cr.home_dir, cr.file_name, NFRAME))
                te_rows.append(_te('Cont', cr.c))
        ctl_len = len(grid)
        for _, wr in g.iterrows():
            grid.append(wp_frames(wr.home_dir, wr.file_name, NFRAME))
            te_rows.append(_te(str(r0.antimicrobial), wr.c))
        if any(not any(f) for f in grid):
            miss_img += 1
        rows.append(dict(
            project_id=str(r0.project_id), sample_id=str(r0.sample_id),
            antimicrobial=str(r0.antimicrobial),
            microbial_id=str(r0.microbial_id), organism_group=str(r0.genus),
            concentration_list=','.join(concs),
            control_len=ctl_len,
            bmd_mic='' if pd.isna(r0.bmd_mic) else str(r0.bmd_mic),
            bmd_mic_order=mic_order(r0.bmd_mic, concs),
            png_frames=json.dumps(grid),
            te_rows=json.dumps(te_rows),
            sample_dir_id=str(r0.sample_dir_id),
        ))
    R = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    p = f'{OUTDIR}/d170_label_source.parquet'
    R.to_parquet(p, index=False)
    print(f'[src25] → {p} · 패널 {len(R):,} · 이미지 결측 의심 {miss_img}', flush=True)
    print(R[['sample_id', 'antimicrobial', 'concentration_list', 'control_len',
             'bmd_mic', 'bmd_mic_order']].head(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
