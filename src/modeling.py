"""Reproducible holdout comparison using six months of account history."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import FEATURES, STATUS, TARGET

SEED = 42


def fit_models(frame):
    """Return fitted pipelines, held-out rows and aligned raw PD estimates."""
    train, test = train_test_split(frame, test_size=0.20, random_state=SEED,
                                   stratify=frame[TARGET])
    amounts = [column for column in FEATURES if column not in STATUS]
    preprocessing = ColumnTransformer([
        ("amount", StandardScaler(), amounts),
        ("status", OneHotEncoder(handle_unknown="ignore", drop=[-1] * len(STATUS)), STATUS),
    ], verbose_feature_names_out=False)
    models = {
        "Logistic regression": make_pipeline(preprocessing, LogisticRegression(max_iter=1000)),
        "Random forest": RandomForestClassifier(n_estimators=160, max_depth=10,
                                                min_samples_leaf=20, random_state=SEED),
    }
    probabilities = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        probabilities[name] = model.predict_proba(test[FEATURES])[:, 1]
    return models, test, probabilities


def threshold_counts(actual, pd_estimate, threshold):
    tn, fp, fn, tp = confusion_matrix(actual, pd_estimate >= threshold, labels=[0, 1]).ravel()
    return dict(zip(("TN", "FP", "FN", "TP"), map(int, (tn, fp, fn, tp))))


def evaluate(actual, probabilities):
    rows = []
    for name, pd_estimate in probabilities.items():
        counts = threshold_counts(actual, pd_estimate, 0.5)
        tn, fp, fn, tp = (counts[key] for key in ("TN", "FP", "FN", "TP"))
        rows.append({
            "Model": name, "Accuracy": (tp + tn) / len(actual),
            "Precision": tp / (tp + fp) if tp + fp else 0,
            "Recall": tp / (tp + fn) if tp + fn else 0,
            "ROC-AUC": roc_auc_score(actual, pd_estimate),
            "Brier": brier_score_loss(actual, pd_estimate), **counts,
        })
    return pd.DataFrame(rows).set_index("Model")


def calibration_points(actual, pd_estimate):
    observed, predicted = calibration_curve(actual, pd_estimate, n_bins=10, strategy="quantile")
    return pd.DataFrame({"Mean predicted PD": predicted, "Observed default rate": observed})


def model_drivers(models, name):
    model = models[name]
    if name == "Logistic regression":
        values = pd.Series(model[-1].coef_[0], index=model[0].get_feature_names_out())
        return values.reindex(values.abs().nlargest(12).index).sort_values()
    return pd.Series(model.feature_importances_, index=FEATURES).nlargest(12).sort_values()


def scenario_losses(test, pd_estimate, lgd, exposure_share):
    # September bill is an imperfect exposure proxy. Negative balances carry zero exposure.
    ead = test["BILL_AMT1"].clip(lower=0).to_numpy() * exposure_share
    return np.asarray(pd_estimate) * lgd * ead

