"""Streamlit entry point for Credit Intelligence Lab."""
from zipfile import BadZipFile

import streamlit as st

from src.dashboard import findings, overview, prediction, simulator
from src.data import load_portfolio
from src.modeling import fit_models

st.set_page_config(page_title="Credit Intelligence Lab", page_icon="◈", layout="wide")
st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at 90% 0%, #25143e 0%, #0D0B1E 44%); }
.block-container { max-width: 1360px; padding-top: 2.5rem; }
.hero { border: 1px solid #42305d; border-radius: 18px; padding: 1.6rem 2rem;
        margin-bottom: 1.5rem; background: linear-gradient(110deg,#1b1533,#251640); }
.hero small { color: #F0ABFC; letter-spacing: .15em; font-weight: 650; }
.hero h1 { margin: .35rem 0; padding: 0; color: #f4edff; }
.hero p { margin: 0; color: #d2c6e9; }
[data-testid="stMetric"] { background: #1b1533; border: 1px solid #3e2e57;
    border-radius: 12px; padding: .8rem 1rem; }
[data-testid="stSidebar"] { border-right: 1px solid #34264f; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading the UCI portfolio…")
def portfolio():
    return load_portfolio()


@st.cache_resource(show_spinner="Fitting models on the training cohort…")
def fitted_portfolio():
    return fit_models(portfolio())


with st.sidebar:
    st.title("Credit Intelligence Lab")
    page = st.radio("Dashboard pages", [
        "Portfolio Overview", "Default Prediction",
        "Credit Risk Simulator", "Findings and Limitations",
    ])
    st.divider()
    st.caption("UCI Default of Credit Card Clients · Taiwan, 2005")
    st.caption("Historical portfolio research · NT$")

descriptions = {
    "Portfolio Overview": "Explore how observed default varies across existing cardholders.",
    "Default Prediction": "Compare two models on the same held-out accounts.",
    "Credit Risk Simulator": "Test how review thresholds and exposure assumptions affect the portfolio.",
    "Findings and Limitations": "Interpret the evidence and its boundaries for credit policy.",
}
st.markdown(f"""
<div class="hero"><small>CREDIT INTELLIGENCE LAB</small>
<h1>{page}</h1><p>{descriptions[page]}</p></div>
""", unsafe_allow_html=True)

try:
    frame = portfolio()
except (OSError, ValueError, RuntimeError, BadZipFile) as error:
    st.error(f"Unable to load the UCI portfolio. See the README for offline setup. ({error})")
    st.stop()

if page == "Portfolio Overview":
    overview(frame)
else:
    models, test, probabilities = fitted_portfolio()
    if page == "Default Prediction":
        prediction(models, test, probabilities)
    elif page == "Credit Risk Simulator":
        simulator(test, probabilities)
    else:
        findings(frame, test, probabilities)

