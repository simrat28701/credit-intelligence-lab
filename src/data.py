"""Load the original UCI workbook and reproduce notebook portfolio bands."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "uci_credit_cards.zip"
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET = "DEFAULT_NEXT_MONTH"
STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
FEATURES = (
    ["LIMIT_BAL"] + STATUS
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
LIMIT_LABELS = ["<50k", "50–<100k", "100–<200k", "200–<300k", "≥300k"]
AGE_LABELS = ["21–29", "30–39", "40–49", "50–59", "60+"]
REPAYMENT_LABELS = ["No positive delay code", "1 month late", "2+ months late"]


def load_portfolio():
    """Cache the validated official archive for offline reruns."""
    cached = ARCHIVE.exists()
    if cached:
        source = ARCHIVE.read_bytes()
    else:
        with urlopen(URL, timeout=45) as response:
            source = response.read()
    with ZipFile(BytesIO(source)) as archive:
        workbook = archive.read("default of credit card clients.xls")
    frame = pd.read_excel(BytesIO(workbook), header=1, engine="xlrd")
    frame = frame.rename(columns={"default payment next month": TARGET})
    expected = {"ID", "SEX", "EDUCATION", "MARRIAGE", "AGE", TARGET, *FEATURES}
    if (frame.shape != (30_000, 25) or set(frame.columns) != expected
            or frame.isna().any().any() or not frame["ID"].is_unique
            or set(frame[TARGET]) != {0, 1}):
        raise ValueError("Unexpected UCI workbook schema or values.")
    if not cached:
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(source)

    frame["Credit band"] = pd.cut(frame["LIMIT_BAL"],
        [0, 50_000, 100_000, 200_000, 300_000, float("inf")],
        labels=LIMIT_LABELS, right=False)
    frame["Age band"] = pd.cut(frame["AGE"],
        [21, 30, 40, 50, 60, float("inf")], labels=AGE_LABELS, right=False)
    frame["September status"] = pd.cut(frame["PAY_0"],
        [-float("inf"), 1, 2, float("inf")], labels=REPAYMENT_LABELS, right=False)
    return frame


def segment_rates(frame, dimension):
    return (frame.groupby(dimension, observed=True)[TARGET]
            .agg(customers="size", default_rate="mean").reset_index())


def sql_segments(frame):
    with duckdb.connect() as connection:
        connection.register("portfolio", frame)
        return connection.execute(
            (ROOT / "sql" / "portfolio_segments.sql").read_text(encoding="utf-8-sig")
        ).df()

