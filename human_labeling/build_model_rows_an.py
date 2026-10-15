# -*- coding: utf-8 -*-
"""★라벨링 도구용 모델 판정 배지 — FDA analytical 8셋 (2026-10-15 사용자 지시).

`build_model_rows.py` 가 d170 을 처리하는 것과 **같은 규약**으로 analytical 셋을 붙인다.
`MODEL_ROWS` 는 소스 구분 없이 pk 로만 조회하므로 같은 `model_rows.json` 에 넣으면
app.py 를 고치지 않고 배지가 뜬다(pk 에 project_id 가 있어 d170 과 충돌하지 않는다).

  pk = "<project_id>|<sample_id>|<antimicrobial>|<sample_dir_id>"
  값 = {cdn: [농도별], cdn_mic, op_mic, nG: [농도별 G 표수], nstruct}

★값 규약은 **`gng` 그대로** — `"0"`=G(자람) · `"1"`=NG. app.py `_g()` 가 그 규약으로 읽는다.
  ⚠analytical parquet 의 `pred_gng` 는 **`G`/`NG` 문자**다 → 변환해서 넣는다.

★무엇을 어디에 넣나
  · `cdn`     = 그 셋의 **route 별 채택 모델**(배포 라우팅) 6h 판정. 패널당 모델은 1개다.
  · `op_mic`  = 장비 운영 판정(`drast_mic`). analytical 은 운영 gng 가 없어 MIC 만 넣는다.
  · `nG`      = **구조 비교본**이 G 로 본 수. `_6h__*` 산출물이 있는 셋만(현재 GP 2셋).
                EN·GN 셋은 구조 추론이 아직 없다 → `nstruct` 가 빠진 채로 뜬다.
  · `dyn_*`   = **EN route 만**(2026-10-15 사용자 지시 "en계열이면 dynamic결과도 보여줘").
                `{ds}_dyn.parquet` 에서 MIC·SIR·판정시각과 **시점별 MIC**(3/4/5/6h)를 넣는다.
                ⚠AC·PA 는 같은 파일에 있어도 **dynamic 이 아니다** — 6h 고정이라
                  `mic_4h` 가 100% 결측이고 `decision_time` 도 6.7 뿐이다. 그래서 EN 만 넣는다.
                ⚠`decision_time` 은 **준비시간 포함**이다(4.7=t4 · 5.7=t5 · 6.7=t6).
                ⚠dyn 산출물의 `pred_gng` 는 비어 있다 → 농도 칸 배지는 못 만들고 MIC 만 넣는다.

⚠**소스 parquet 을 건드리지 않는다.** parquet 을 바꾸면 idx 가 밀려 열어 둔 탭이 깨진다
  (2026-09-08 사고). 배지는 별도 파일로만 붙인다.
⚠기존 `model_rows.json`(d170 분)을 **덮지 않고 merge** 한다.

사용: python build_model_rows_an.py
"""
import os
import sys
import glob
import json
import shutil

import pandas as pd

D25 = '/home/kptae/project/drast_25_lrcn'
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(D25, 'newmodel/output/analytical25')
SRC_DIR = '/home/kptae/data/allinfo/d25'

# 라벨 소스 ↔ analytical 산출물 이름
SETS = {
    'an_s26_intf':    'S26_Interfering_Substances',
    'an_e27_intf':    'E27_Interfering_Substances',
    'an_s27_intf':    'S27_Interfering_Substances',
    'an_gn_s26_stab': 'GN_S26_Sample_stability',
    'an_gn_e27_stab': 'GN_E27_Sample_stability',
    'an_gp_s27_stab': 'GP_S27_Sample_stability',
    'an_s26_bcbe':    'S26_Blood_Culture_Bottle_Equivalency',
    'an_e27_bcbe':    'E27_Blood_Culture_Bottle_Equivalency',
}
K4 = ['project_id', 'sample_id', 'antimicrobial', 'sample_dir_id']


def pk(df):
    return df[K4].astype(str).agg('|'.join, axis=1)


def gng01(s):
    """`G,NG,NG` → `['0','1','1']`. 규약은 app.py `_g()` 와 같다."""
    out = []
    for x in str(s).split(','):
        x = x.strip()
        if not x:
            continue
        out.append('0' if x == 'G' else '1')
    return out


def main():
    rows = {}
    stat = []
    for sid, ds in SETS.items():
        src_p = f'{SRC_DIR}/{sid}_label_source.parquet'
        base_p = f'{AN}/{ds}_6h.parquet'
        if not os.path.exists(src_p):
            stat.append((sid, 0, 0, 0, 0, '소스 없음'))
            continue
        if not os.path.exists(base_p):
            stat.append((sid, 0, 0, 0, 0, '6h 추론 없음'))
            continue
        src = pd.read_parquet(src_p, columns=K4)
        want = set(pk(src))

        d = pd.read_parquet(base_p)
        d['pk'] = pk(d)
        d = d[d.pk.isin(want)]

        # ★구조 비교본 — `{ds}_6h__*.parquet`. 기본 모델도 섞여 들어오므로 모델명으로 모은다.
        var = sorted(glob.glob(f'{AN}/{ds}_6h__*.parquet'))
        V = {}
        for f in var:
            if f.endswith('__ab_selfcheck.parquet'):
                continue          # 자기검증본 — 모델 비교가 아니다
            x = pd.read_parquet(f)
            x['pk'] = pk(x)
            x = x[x.pk.isin(want)]
            for m, g in x.groupby(x.model.astype(str)):
                V.setdefault(m, []).append(g[['pk', 'pred_gng']])
        VM = {m: pd.concat(v, ignore_index=True).drop_duplicates('pk').set_index('pk')
              for m, v in V.items()}

        # ★dynamic — EN route 만. AC·PA 는 같은 파일에 있어도 6h 고정이라 뺀다.
        DY = None
        dyn_p = f'{AN}/{ds}_dyn.parquet'
        if os.path.exists(dyn_p):
            y = pd.read_parquet(dyn_p)
            y = y[y.route.astype(str) == 'EN']
            if len(y):
                y['pk'] = pk(y)
                DY = y[y.pk.isin(want)].drop_duplicates('pk').set_index('pk')

        n_cdn = n_struct_panel = n_dyn = 0
        for p, g in d.groupby('pk'):
            r = {}
            r0 = g.iloc[0]
            r['cdn'] = gng01(r0.pred_gng)
            r['cdn_mic'] = str(r0.pred_mic)
            if pd.notna(r0.get('drast_mic')):
                r['op_mic'] = str(r0.drast_mic)
            n_cdn += 1
            seqs = []
            for m, t in VM.items():
                if p in t.index:
                    seqs.append(gng01(t.loc[p].pred_gng))
            if seqs:
                n = max(len(s) for s in seqs)
                r['nG'] = [int(sum(1 for s in seqs if i < len(s) and s[i] == '0'))
                           for i in range(n)]
                r['nstruct'] = len(seqs)
                n_struct_panel += 1
            if DY is not None and p in DY.index:
                y = DY.loc[p]
                r['dyn_mic'] = str(y.pred_mic)
                if pd.notna(y.get('pred_sir_dyn')):
                    r['dyn_sir'] = str(y.pred_sir_dyn)
                if pd.notna(y.get('decision_time')):
                    r['dyn_t'] = float(y.decision_time)
                # 시점별 MIC — "조기에 무엇이 달랐나"를 라벨러가 바로 본다
                st = {}
                for t in ('3h', '4h', '5h', '6h'):
                    v = y.get(f'mic_{t}')
                    if pd.notna(v):
                        st[t] = str(v)
                if st:
                    r['dyn_steps'] = st
                n_dyn += 1
            rows[p] = r
        note = f'구조 {len(VM)}종' if VM else '구조 추론 없음'
        stat.append((sid, len(want), n_cdn, n_struct_panel, n_dyn, note))

    # ── 기존(d170) 과 merge
    p = os.path.join(HERE, 'model_rows.json')
    old = {}
    if os.path.exists(p):
        shutil.copy(p, p + '.bak')
        with open(p, encoding='utf-8') as f:
            old = json.load(f)
    merged = dict(old)
    merged.update(rows)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f'{"셋":18s}{"소스":>8s}{"배지":>8s}{"구조배지":>10s}{"dyn(EN)":>10s}  비고')
    print('-' * 80)
    for sid, w, c, s, y, note in stat:
        print(f'{sid:18s}{w:8,}{c:8,}{s:10,}{y:10,}  {note}')
    print(f'\n기존 {len(old):,} + 신규 {len(rows):,} → 합계 {len(merged):,}패널')
    print(f'→ {p} (백업 {os.path.basename(p)}.bak)')
    print('⚠app.py 는 기동 시 1회만 읽는다 — 서버 재시작 필요')


if __name__ == '__main__':
    main()
