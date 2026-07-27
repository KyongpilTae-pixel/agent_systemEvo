#!/bin/bash
# technical_error 추론 — 우리쪽 관리본(2026-07-27 이관). 4-GPU DDP.
#   모델/입력/출력 = /home/kptae/data/technical_error (TECH_ERR_DATA 로 override 가능).
#   ★ 4장 모두 사용 → 학습(ctl_traj 등)과 충돌하므로 GPU 여유 시 실행.
set -e
source /home/kptae/miniconda3/etc/profile.d/conda.sh
conda activate qnt_algorithm
cd "$(dirname "$(readlink -f "$0")")"
python predict.py
