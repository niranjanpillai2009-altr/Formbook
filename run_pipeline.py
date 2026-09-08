"""One-shot pipeline: download data -> build features -> train -> evaluate.

    python run_pipeline.py                 # full run (Optuna tuning, ~30-40 min on a laptop)
    python run_pipeline.py --quick         # reuse saved XGBoost params (~8 min)
    python run_pipeline.py --refresh-data  # re-download the latest seasons first
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from epl_predictor import evaluate, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip Optuna tuning, reuse models/xgb_params.json")
    ap.add_argument("--trials", type=int, default=35, help="Optuna trials when tuning")
    ap.add_argument("--refresh-data", action="store_true", help="re-download the latest seasons")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    train.run(n_trials=args.trials, refresh=True, refresh_data=args.refresh_data, skip_tuning=args.quick)
    evaluate.run()
    print("\nDone. Figures are in reports/figures/, the model in models/final_model.joblib.")
    print('Try:  python -m epl_predictor.predict "Arsenal" "Chelsea"')


if __name__ == "__main__":
    main()
