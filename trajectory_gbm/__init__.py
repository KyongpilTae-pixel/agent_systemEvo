"""trajectory_gbm — brightness·object_area 등 trajectory 신호 GBM 통합 패키지.

brightness 와 object_area 는 입력 trajectory 컬럼만 다르고 feature 추출·GBM 학습·
GroupKFold CV·비대칭 보정 로직이 동일하다. 그 공통 코어를 한 곳에 모았다.

  from trajectory_gbm import config, model, corrector
  sig = config.get_signal("object_area")
  meta = model.train_crossdomain(sig)            # TRAIN→FDA 학습
  out, df = model.cv_auroc_indomain(sig)         # GroupKFold OOF AUROC
"""
from . import config, features, model, corrector  # noqa: F401
from .config import SIGNALS, SignalConfig, get_signal  # noqa: F401
