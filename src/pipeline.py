"""Run with python -m src.pipeline to reproduce the dashboard's reference results."""
import json

import numpy as np

from src.data import ROOT, TARGET, load_portfolio, segment_rates, sql_segments
from src.modeling import evaluate, fit_models, scenario_losses, threshold_counts


def main():
    frame = load_portfolio()
    segments = sql_segments(frame)
    assert segments["customers"].sum() == len(frame)
    assert segments["observed_defaults"].sum() == frame[TARGET].sum()
    _, test, probabilities = fit_models(frame)
    logistic = probabilities["Logistic regression"]
    losses = scenario_losses(test, logistic, 0.45, 1.0)
    results = {
        "customers": len(frame),
        "defaults": int(frame[TARGET].sum()),
        "default_rate": frame[TARGET].mean(),
        "test_customers": len(test),
        "majority_accuracy": 1 - test[TARGET].mean(),
        "segments": {
            column: segment_rates(frame, column).to_dict(orient="records")
            for column in ("Credit band", "Age band", "September status")
        },
        "models": evaluate(test[TARGET], probabilities).to_dict(orient="index"),
        "logistic_scenarios": {
            str(threshold): {
                **threshold_counts(test[TARGET], logistic, threshold),
                "flagged_ecl_ntd": float(losses[logistic >= threshold].sum()),
            }
            for threshold in (0.2, 0.3, 0.5)
        },
        "test_ecl_ntd": float(np.sum(losses)),
    }
    output = json.dumps(results, indent=2, ensure_ascii=False)
    (ROOT / "data" / "results.json").write_text(output + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

