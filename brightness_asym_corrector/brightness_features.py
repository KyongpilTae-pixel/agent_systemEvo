"""[SHIM] brightness trajectory feature 추출 → trajectory_gbm.features 로 통합.

2026-06-04: brightness·object_area 가 입력 컬럼만 다르고 feature 로직이 동일함을 확인,
공통 코어를 `agent_system/trajectory_gbm/features.py` 로 일반화. 이 파일은 기존 import
(`from brightness_features import build_features` 등)을 깨지 않기 위한 re-export shim.
"""
from __future__ import annotations

import sys
from pathlib import Path

# repo root 를 path 에 올려 패키지 import 가능하게
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_system.trajectory_gbm.features import (  # noqa: E402,F401
    MODEL_TIME_LEN, CTL_PICK_INDEX, EPS, FEATURE_PREFIX,
    parse_csv_list, features_for_row, feature_columns, build_features,
)
