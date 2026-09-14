# -*- coding: utf-8 -*-
"""★라벨링 도구용 모델 판정 배지 생성 (2026-09-11).

라벨러가 "이 칸을 모델들은 어떻게 봤나"를 화면에서 바로 보게 한다.

  `model_rows.json` : `"<project_id>|<sample_id>|<antimicrobial>"` →
      {cdn: [농도별], cdn_mic, op: [농도별], op_mic, nG: [농도별 G 표수], nstruct}
  ★값 규약은 **`gng` 그대로** — `"0"`=G(자람) · `"1"`=NG. app.py `_g()` 가 그 규약으로 읽는다.

⚠**소스 parquet 을 건드리지 않는다.** parquet 을 바꾸면 idx 가 밀려 열어 둔 탭이 깨진다
  (2026-09-08 사고). 그래서 배지는 **별도 파일**로 붙인다.

⚠기존 `model_rows.json`(2026-09-08)은 **602패널분**이라 소스 5,877 중 10% 만 배지가 떴다.
  소스를 재생성할 때마다 이 스크립트도 같이 돌려야 한다.

사용: python build_model_rows.py
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

D25 = '/home/kptae/project/drast_25_lrcn'
sys.path.insert(0, D25)
sys.path.insert(0, '/home/kptae/project/qnt_algorithm')
os.chdir(D25)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = 'newmodel/output'
SRC = '/home/kptae/data/allinfo/d25/d170_label_source.parquet'
CDN = 'lightpanel25cdn_base_center_6h__f180__train_deploy_en'
OPN = 'deployed_lrcn_gng'


def _list(v):
    """parquet 의 `gng`/`concs` 는 문자 분할 저장 — 이어붙인 뒤 쉼표로 쪼갠다."""
    try:
        s = ''.join(str(x) for x in v)
    except TypeError:
        s = str(v)
    return [x.strip() for x in s.split(',') if x.strip() != '']


def main():
    # ★`sample_dir_id` 까지 맞춘다(2026-09-11). 같은 (project,sample,drug) 가 **두 번 시험**된
    #   검체가 있다(접종량 차이 등). 3중 키로 묶으면 두 시험의 판정이 **한 패널로 합쳐져**
    #   `nstruct` 가 36(18×2)이 되고 농도 칸이 절반으로 줄어든다.
    #   소스는 3중 키당 1행이며 어느 시험인지 `sample_dir_id` 로 특정한다 — 그것과 짝을 맞춘다.
    src = pd.read_parquet(SRC, columns=['project_id', 'sample_id', 'antimicrobial',
                                        'sample_dir_id'])
    # ★키에 `sample_dir_id` 를 넣는다 — 3중 키면 분리된 두 시험 중 한쪽만 배지가 뜬다.
    src['pk'] = (src.project_id.astype(str) + '|' + src.sample_id.astype(str) + '|'
                 + src.antimicrobial.astype(str) + '|' + src.sample_dir_id.astype(str))
    want = set(src.pk)
    print(f'[rows] 소스 패널 {len(want):,}', flush=True)

    # ★기본을 **정본 추론 산출물**로 둔다(2026-09-14). 옛 `pred_full_g*` 는 정본 전환
    #   이전(원본 CSV · 958검체) 결과라 라벨링 화면의 모델 배지가 **지금 모델과 다른 판정**을
    #   보여준다. 환경변수 `ROWS_PRED` 로 glob 패턴을 바꿀 수 있다(쉼표 구분).
    _pat = os.environ.get('ROWS_PRED',
                          f'{OUT}/pred_canon_*.parquet,{OUT}/pred_d170_baseline_canon.parquet')
    fs = []
    for g in _pat.split(','):
        fs += sorted(glob.glob(g.strip()))
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        raise SystemExit(f'[rows] 추론 parquet 없음 — 패턴 {_pat}')
    print(f'[rows] 입력 {len(fs)}파일: ' + ', '.join(os.path.basename(f) for f in fs), flush=True)
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    D['pk'] = (D.project_id.astype(str) + '|' + D.sample_id.astype(str) + '|'
               + D.antimicrobial.astype(str) + '|' + D.sample_dir_id.astype(str))
    D = D[D.pk.isin(want)]
    struct = sorted(m for m in set(D.model.astype(str)) if m != OPN)
    print(f'[rows] 대상 행 {len(D):,} · 구조모델 {len(struct)}', flush=True)

    out = {}
    for pk, g in D.groupby('pk'):
        r = {}
        cd = g[g.model == CDN]
        if len(cd):
            r['cdn'] = _list(cd.gng.iloc[0])
            r['cdn_mic'] = str(cd.pred_mic_s.iloc[0])
        op = g[g.model == OPN]
        if len(op):
            r['op'] = _list(op.gng.iloc[0])
            r['op_mic'] = str(op.pred_mic_s.iloc[0])
        # ★구조모델 18종 중 **G(=`0`) 로 본 수**를 농도별로 센다 — "대부분 NG 로 봤나"를 한눈에.
        st = g[g.model.isin(struct)]
        if len(st):
            seqs = [_list(x) for x in st.gng]
            n = max(len(s) for s in seqs)
            r['nG'] = [int(sum(1 for s in seqs if i < len(s) and s[i] == '0'))
                       for i in range(n)]
            r['nstruct'] = len(seqs)
        if r:
            out[pk] = r

    p = os.path.join(HERE, 'model_rows.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    cov = len(set(out) & want)
    print(f'[rows] → {p} · {len(out):,}패널 · 소스 커버 {cov:,}/{len(want):,} '
          f'({cov / len(want) * 100:.1f}%)', flush=True)
    miss = len(want) - cov
    if miss:
        print(f'[rows] ⚠예측 없는 소스 패널 {miss:,} — 배지 없이 뜬다', flush=True)


if __name__ == '__main__':
    main()
