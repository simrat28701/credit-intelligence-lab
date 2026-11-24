"""Dashboard rendering. Computation lives in data.py and modeling.py."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import AGE_LABELS, LIMIT_LABELS, REPAYMENT_LABELS, TARGET, segment_rates, sql_segments
from src.modeling import calibration_points, evaluate, model_drivers, scenario_losses, threshold_counts

LAVENDER = "#B69CFF"
PINK = "#F0ABFC"


def chart(figure, percent=False):
    figure.update_layout(
        paper_bgcolor="#171229", plot_bgcolor="#171229", font_color="#F3ECFF",
        margin=dict(l=24, r=24, t=24, b=28), height=330,
        xaxis_gridcolor="#352849", yaxis_gridcolor="#352849",
    )
    if percent:
        figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(figure, theme=None, width="stretch", config={"displayModeBar": False})


def metric_row(items):
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value, border=True)


def overview(frame):
    credit, age, status = st.columns(3)
    limits = credit.multiselect("Credit limit bands (NT$)", LIMIT_LABELS)
    ages = age.multiselect("Age bands", AGE_LABELS)
    statuses = status.multiselect("September repayment status", REPAYMENT_LABELS)
    st.caption("Leave a filter empty to include all bands.")
    selected = frame[
        frame["Credit band"].isin(limits or LIMIT_LABELS)
        & frame["Age band"].isin(ages or AGE_LABELS)
        & frame["September status"].isin(statuses or REPAYMENT_LABELS)
    ]
    if selected.empty:
        st.info("No customers match these filters. Choose a wider cohort.")
        return
    metric_row([
        ("Customers", f"{len(selected):,}"),
        ("Observed defaults", f"{selected[TARGET].sum():,}"),
        ("Default rate", f"{selected[TARGET].mean():.2%}"),
        ("Median credit limit", f"NT$ {selected['LIMIT_BAL'].median():,.0f}"),
    ])
    left, right = st.columns(2)
    with left:
        st.subheader("Credit limit distribution")
        chart(px.histogram(selected, x="LIMIT_BAL", nbins=35,
                           labels={"LIMIT_BAL": "Credit limit (NT$)"},
                           color_discrete_sequence=[LAVENDER]))
    with right:
        rate_chart(selected, "Credit band", LIMIT_LABELS, "Default by credit limit")
    left, right = st.columns(2)
    with left:
        rate_chart(selected, "Age band", AGE_LABELS, "Default by age")
    with right:
        rate_chart(selected, "September status", REPAYMENT_LABELS, "Default by repayment history")

    st.subheader("Portfolio segments")
    with st.expander("View the DuckDB segmentation", expanded=True):
        st.caption("Credit band × September repayment status · current filtered cohort")
        st.dataframe(sql_segments(selected).style.format({
            "default_rate": "{:.2%}", "median_limit_ntd": "{:,.0f}"
        }), hide_index=True, width="stretch")
    st.info("In the full portfolio, recent late payments are associated with higher observed "
            "default rates. Credit limits reflect earlier underwriting and customer circumstances. "
            "Age patterns may also reflect differences in account history and portfolio composition.")


def rate_chart(frame, dimension, order, title):
    st.subheader(title)
    rates = segment_rates(frame, dimension)
    figure = px.bar(rates, x=dimension, y="default_rate", hover_data=["customers"],
                    category_orders={dimension: order}, color_discrete_sequence=[PINK],
                    labels={"default_rate": "Observed default rate"})
    chart(figure, percent=True)


def prediction(models, test, probabilities):
    actual = test[TARGET]
    results = evaluate(actual, probabilities)
    baseline = 1 - actual.mean()
    metric_row([
        ("Holdout customers", f"{len(test):,}"),
        ("Holdout default rate", f"{actual.mean():.2%}"),
        ("Always non-default accuracy", f"{baseline:.2%}"),
        ("Always non-default recall", "0%"),
    ])
    st.subheader("Holdout performance")
    st.caption("Accuracy, precision and recall use a 50% threshold. ROC-AUC measures ranking. "
               "Brier score measures squared probability error, with lower values preferred.")
    st.dataframe(results.iloc[:, :5].style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
        "ROC-AUC": "{:.3f}", "Brier": "{:.4f}",
    }), width="stretch")
    st.info("Most accounts did not default. Accuracy can look high even when a model misses "
            "many defaults. Recall measures how many observed defaults a policy catches.")

    left, right = st.columns(2)
    with left:
        st.subheader("Probability calibration")
        figure = go.Figure()
        figure.add_scatter(x=[0, 1], y=[0, 1], name="Perfect calibration",
                           line=dict(color="#888099", dash="dot"))
        for (name, scores), color in zip(probabilities.items(), [LAVENDER, PINK]):
            points = calibration_points(actual, scores)
            figure.add_scatter(x=points.iloc[:, 0], y=points.iloc[:, 1],
                               mode="lines+markers", name=name, line=dict(color=color))
        figure.update_layout(xaxis_title="Mean predicted PD", yaxis_title="Observed default rate",
                             legend=dict(orientation="h", y=-0.25))
        figure.update_xaxes(tickformat=".0%")
        chart(figure, percent=True)
        st.caption("Up to ten equal-frequency bins per model. Points above the diagonal indicate underestimated risk.")
    with right:
        st.subheader("Confusion matrices")
        for name, row in results.iterrows():
            st.markdown(f"**{name}**")
            matrix = pd.DataFrame([[int(row.TN), int(row.FP)], [int(row.FN), int(row.TP)]],
                                  index=["Observed non-default", "Observed default"],
                                  columns=["Unflagged", "Flagged"])
            st.dataframe(matrix, width="stretch")

    st.subheader("Model drivers")
    model_name = st.segmented_control("Model explanation", list(models), default="Logistic regression")
    if model_name:
        values = model_drivers(models, model_name)
        logistic = model_name == "Logistic regression"
        label = "Log-odds coefficient" if logistic else "Impurity importance"
        figure = px.bar(x=values.to_numpy(), y=values.index, orientation="h",
                        labels={"x": label, "y": "Feature"},
                        color_discrete_sequence=[LAVENDER if logistic else PINK])
        chart(figure)
        st.caption("12 largest absolute coefficients. Amounts are per training standard deviation. "
                   "Repayment codes compare with −1 for the same month, holding other terms fixed."
                   if logistic else
                   "12 largest training impurity importances. Correlated features can split credit "
                   "for a signal, and continuous features offer more candidate splits.")


def simulator(test, probabilities):
    model, cutoff, severity, exposure = st.columns([1.2, 1, 1, 1])
    name = model.selectbox("PD model", list(probabilities))
    threshold = cutoff.slider("Flag at PD (%)", 0, 100, 30) / 100
    lgd = severity.slider("LGD (%)", 0, 100, 45) / 100
    exposure_share = exposure.slider("Balance used as EAD (%)", 0, 200, 100, 5) / 100
    actual = test[TARGET]
    scores = probabilities[name]
    flagged = scores >= threshold
    counts = threshold_counts(actual, scores, threshold)
    losses = scenario_losses(test, scores, lgd, exposure_share)
    metric_row([
        ("Flagged accounts", f"{flagged.sum():,} / {len(test):,}"),
        ("False positives", f"{counts['FP']:,}"),
        ("False negatives", f"{counts['FN']:,}"),
        ("Observed defaults flagged", f"{counts['TP'] / actual.sum():.1%}"),
    ])
    st.subheader("Illustrative one-month expected credit loss")
    metric_row([
        ("All holdout accounts", f"NT$ {losses.sum():,.0f}"),
        ("Flagged accounts", f"NT$ {losses[flagged].sum():,.0f}"),
        ("Unflagged accounts", f"NT$ {losses[~flagged].sum():,.0f}"),
    ])
    st.caption("Per account: model PD × assumed LGD × assumed share of positive September "
               "statement balance. Moving the threshold reallocates this estimate across groups.")
    st.subheader("Review workload and missed defaults")
    sweep = []
    for candidate in sorted({threshold, *(i / 100 for i in range(0, 101, 2))}):
        row = threshold_counts(actual, scores, candidate)
        sweep.append({"Threshold": candidate, "Flagged": row["TP"] + row["FP"],
                      "False positives": row["FP"], "False negatives": row["FN"]})
    figure = px.line(pd.DataFrame(sweep), x="Threshold", y=["Flagged", "False positives", "False negatives"],
                     color_discrete_sequence=[LAVENDER, PINK, "#72DACD"],
                     labels={"value": "Accounts", "variable": "Outcome"})
    figure.add_vline(x=threshold, line_dash="dot", line_color="#DFD0F3")
    figure.update_xaxes(tickformat=".0%")
    figure.update_layout(legend=dict(orientation="h", y=-0.25))
    chart(figure)
    st.info("A lower threshold can catch more potential defaults, at the cost of more reviews "
            "and possible friction for customers who would repay. A higher threshold reduces "
            "review workload and can leave more risk outside review. The dataset contains no "
            "measured effect of intervention, so flagged ECL measures exposure to risk in that group.")


def findings(frame, test, probabilities):
    actual = test[TARGET]
    results = evaluate(actual, probabilities)
    status = segment_rates(frame, "September status").set_index("September status")
    limits = segment_rates(frame, "Credit band").set_index("Credit band")
    ages = segment_rates(frame, "Age band").set_index("Age band")
    st.subheader("What the data shows")
    st.markdown(
        f"- **{frame[TARGET].mean():.2%}** of **{len(frame):,}** cardholders defaulted the following month.\n"
        f"- Default was **{status.loc[REPAYMENT_LABELS[2], 'default_rate']:.1%}** for accounts coded "
        f"two or more months late in September and **{status.loc[REPAYMENT_LABELS[0], 'default_rate']:.1%}** "
        "for accounts with no positive delay code.\n"
        f"- The observed default rate was **{limits.loc[LIMIT_LABELS[0], 'default_rate']:.1%}** below "
        f"NT$50,000 and **{limits.loc[LIMIT_LABELS[-1], 'default_rate']:.1%}** at NT$300,000 or above.\n"
        f"- The rate was **{ages.loc[AGE_LABELS[1], 'default_rate']:.1%}** for ages 30 to 39 "
        f"and **{ages.loc[AGE_LABELS[-1], 'default_rate']:.1%}** for ages 60 and up, a group of "
        f"**{ages.loc[AGE_LABELS[-1], 'customers']:,}** customers.\n"
        f"- Holdout ROC-AUC was **{results.loc['Logistic regression', 'ROC-AUC']:.3f}** for logistic "
        f"regression and **{results.loc['Random forest', 'ROC-AUC']:.3f}** for the forest."
    )
    st.subheader("How far the evidence goes")
    st.markdown("""
- **Association and causality.** Credit limits reflect earlier underwriting and customer characteristics. These comparisons do not estimate the effect of changing a limit.
- **Existing customers.** The sample covers established accounts from one historical setting. New applicants may lack repayment histories, and rejected applicants' outcomes are unobserved. A random holdout also shares the original time period.
- **Fairness.** Age, sex, education and marital status are excluded from both models. Financial predictors can still act as proxies. An audit would compare subgroup calibration, error rates and sample sizes on representative data.
- **Loss assumptions.** The public dataset lacks realized recoveries, future drawdowns and exposure at default. The simulator uses a statement-balance proxy and user-supplied assumptions.
- **Decision scope.** Use the scenarios to discuss account review capacity and trade-offs. An intervention policy needs evidence about review costs, customer effects and recoverable loss.
""")
    st.markdown("[UCI dataset, documentation and license](https://doi.org/10.24432/C55S3H)")

