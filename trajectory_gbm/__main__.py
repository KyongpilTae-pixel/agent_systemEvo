"""trajectory_gbm CLI.

  python -m agent_system.trajectory_gbm train --signal object_area
  python -m agent_system.trajectory_gbm cv    --signal object_area --folds 5
  python -m agent_system.trajectory_gbm train --signal brightness

train : TRAIN_CSV→FDA cross-domain 학습 (모델 저장 + in/cross AUROC)
cv    : FDA in-domain GroupKFold OOF AUROC (honest baseline)
"""
from __future__ import annotations

import argparse
import json

from . import config as C
from . import model


def main() -> None:
    ap = argparse.ArgumentParser(prog="trajectory_gbm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="cross-domain 학습 (TRAIN→FDA)")
    pt.add_argument("--signal", required=True, choices=list(C.SIGNALS))
    pt.add_argument("--no-save", action="store_true")

    pc = sub.add_parser("cv", help="in-domain GroupKFold OOF AUROC")
    pc.add_argument("--signal", required=True, choices=list(C.SIGNALS))
    pc.add_argument("--folds", type=int, default=5)

    args = ap.parse_args()
    sig = C.get_signal(args.signal)

    if args.cmd == "train":
        meta = model.train_crossdomain(sig, save=not args.no_save)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        if not args.no_save:
            print(f"[saved] {sig.model_path}")
    elif args.cmd == "cv":
        out, _ = model.cv_auroc_indomain(sig, n_splits=args.folds)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
