
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Loan Portfolio Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080d18; color: #e8eef8; }
section[data-testid="stSidebar"] { background: #0d1424; border-right: 1px solid #243149; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -0.6px; }
.hero { padding: 18px 24px; border: 1px solid #243149; border-radius: 18px;
        background: linear-gradient(135deg,#101b30,#0b1220); margin-bottom: 18px; }
.hero-title { font-size: 34px; font-weight: 800; margin: 0; color: #f4f8ff; }
.hero-sub { color: #91a4c2; margin-top: 6px; font-size: 14px; }
.kpi { background: linear-gradient(135deg,#111e33,#0e1729); border: 1px solid #263752;
       border-radius: 16px; padding: 18px 20px; min-height: 115px; }
.kpi-label { color: #93a7c5; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { color: #f4f8ff; font-size: 29px; font-weight: 800; margin-top: 10px; }
.kpi-note { color: #6ee7d0; font-size: 11px; margin-top: 5px; }
.section { font-size: 19px; font-weight: 750; margin: 24px 0 8px; color: #eaf2ff; }
.small-note { color: #8ea2c0; font-size: 12px; }
div[data-testid="stMetric"] { background: #111e33; border: 1px solid #263752; padding: 14px; border-radius: 14px; }
button[kind="secondary"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    candidates = [
    Path("final_loan_portfolio.csv"),
    Path("data/final_loan_portfolio.csv"),
    Path("/content/final_loan_portfolio.csv"),
    Path("/mnt/data/final_loan_portfolio.csv")
]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        uploaded = None
        return None
    for c in ["original_upb","original_interest_rate","credit_score","original_dti",
              "original_ltv","original_cltv","delinquency_rate","serious_delinquency_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["ever_delinquent_flag","ever_seriously_delinquent_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

df = load_data()
if df is None:
    st.error("Dataset not found. Place final_loan_portfolio.csv inside the data folder or upload it below.")
    uploaded = st.file_uploader("Upload final_loan_portfolio.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

# Sidebar filters
st.sidebar.markdown("## ◈ Portfolio Controls")
st.sidebar.caption("Interactive filters update all dashboard views.")

states = sorted(df["property_state"].dropna().astype(str).unique())
risk_values = sorted(df["combined_risk_indicator"].dropna().astype(str).unique())
purpose_values = sorted(df["loan_purpose"].dropna().astype(str).unique())

selected_states = st.sidebar.multiselect("Property state", states, default=[])
selected_risks = st.sidebar.multiselect("Risk indicator", risk_values, default=[])
selected_purposes = st.sidebar.multiselect("Loan purpose code", purpose_values, default=[])

score_range = st.sidebar.slider(
    "Credit score range", 600, 850,
    (int(max(600, np.nanmin(df["credit_score"]))),
     int(min(850, np.nanmax(df["credit_score"]))))
)
dti_max = st.sidebar.slider("Maximum DTI", 0, 65, 65)
ltv_max = st.sidebar.slider("Maximum LTV", 0, 100, 100)

filtered = df.copy()
if selected_states: filtered = filtered[filtered["property_state"].astype(str).isin(selected_states)]
if selected_risks: filtered = filtered[filtered["combined_risk_indicator"].astype(str).isin(selected_risks)]
if selected_purposes: filtered = filtered[filtered["loan_purpose"].astype(str).isin(selected_purposes)]
filtered = filtered[filtered["credit_score"].fillna(0).between(score_range[0], score_range[1]) | filtered["credit_score"].isna()]
filtered = filtered[filtered["original_dti"].fillna(0) <= dti_max]
filtered = filtered[filtered["original_ltv"].fillna(0) <= ltv_max]

def money(x):
    if pd.isna(x): return "—"
    return f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.1f}M"

def pct(x):
    return "—" if pd.isna(x) else f"{x:.2f}%"

total_loans = len(filtered)
portfolio = filtered["original_upb"].sum()
delinq = filtered["ever_delinquent_flag"].sum()
serious = filtered["ever_seriously_delinquent_flag"].sum()
delinq_rate = delinq / total_loans * 100 if total_loans else 0
serious_rate = serious / total_loans * 100 if total_loans else 0

st.markdown("""
<div class="hero">
  <div class="hero-title">Loan Portfolio Intelligence</div>
  <div class="hero-sub">Executive risk analytics • Python + SQL-ready • Interactive portfolio exploration</div>
</div>
""", unsafe_allow_html=True)

kpis = [
    ("Total loans", f"{total_loans:,.0f}", "Filtered portfolio"),
    ("Portfolio value", money(portfolio), "Original UPB • USD"),
    ("Delinquency rate", f"{delinq_rate:.2f}%", "Ever delinquent proxy"),
    ("Serious delinquency", f"{serious_rate:.2f}%", "Status ≥ 3 proxy"),
    ("Average credit score", f"{filtered['credit_score'].mean():.1f}" if len(filtered) else "—", "Available scores"),
]
cols = st.columns(5)
for col, (label, value, note) in zip(cols, kpis):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">Portfolio overview</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    state = (filtered.groupby("property_state", as_index=False)
             .agg(total_loans=("loan_sequence_number","count"),
                  portfolio_value=("original_upb","sum"),
                  delinquency_rate=("ever_delinquent_flag","mean"))
             .sort_values("total_loans", ascending=False).head(15))
    state["delinquency_rate"] *= 100
    fig = px.bar(state, x="total_loans", y="property_state", orientation="h",
                 title="Top states by loan volume", text="total_loans",
                 template="plotly_dark")
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis_title="", xaxis_title="Loans", margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    purpose = (filtered.groupby("loan_purpose", as_index=False)
               .agg(total_loans=("loan_sequence_number","count"),
                    delinquency_rate=("ever_delinquent_flag","mean")))
    purpose["delinquency_rate"] *= 100
    fig = px.bar(purpose, x="loan_purpose", y="total_loans", color="delinquency_rate",
                 title="Loan purpose volume with delinquency intensity",
                 hover_data={"delinquency_rate":":.2f"}, template="plotly_dark")
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Purpose code", yaxis_title="Loans", margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section">Risk intelligence</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    risk = (filtered.groupby("combined_risk_indicator", as_index=False)
            .agg(total_loans=("loan_sequence_number","count"),
                 delinquency_rate=("ever_delinquent_flag","mean"),
                 serious_rate=("ever_seriously_delinquent_flag","mean")))
    risk["delinquency_rate"] *= 100
    risk["serious_rate"] *= 100
    fig = px.bar(risk, x="combined_risk_indicator", y="delinquency_rate",
                 color="combined_risk_indicator", title="Delinquency rate by analytical risk band",
                 hover_data={"total_loans":True,"serious_rate":":.2f"}, template="plotly_dark")
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="", yaxis_title="Delinquency rate (%)", showlegend=False,
                      margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.scatter(filtered.sample(min(7000, len(filtered)), random_state=42),
                     x="original_ltv", y="original_dti",
                     color="credit_risk_category",
                     size="original_upb", hover_data=["property_state","loan_purpose"],
                     title="Exposure map: LTV vs DTI", template="plotly_dark")
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Original LTV (%)", yaxis_title="Original DTI (%)",
                      margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section">Portfolio composition</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    credit = filtered["credit_risk_category"].value_counts(dropna=False).rename_axis("category").reset_index(name="loans")
    fig = px.pie(credit, names="category", values="loans", hole=0.55,
                 title="Credit risk mix", template="plotly_dark")
    fig.update_layout(height=390, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)
with c6:
    dti = filtered["dti_risk_category"].value_counts(dropna=False).rename_axis("category").reset_index(name="loans")
    fig = px.bar(dti, x="category", y="loans", title="DTI risk distribution",
                 color="loans", template="plotly_dark")
    fig.update_layout(height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="", yaxis_title="Loans", margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section">Transparent risk scenario simulator</div>', unsafe_allow_html=True)
st.caption("This is an explainable rule-based screening tool using the project's analytical thresholds. It is not a validated credit underwriting model.")
s1, s2, s3, s4 = st.columns(4)
with s1: sim_score = st.number_input("Credit score", 300, 850, 720)
with s2: sim_dti = st.number_input("DTI (%)", 0.0, 100.0, 35.0)
with s3: sim_ltv = st.number_input("LTV (%)", 0.0, 150.0, 80.0)
with s4: sim_cltv = st.number_input("CLTV (%)", 0.0, 150.0, 80.0)

risk_count = 0
if sim_score < 670: risk_count += 1
if sim_dti > 35: risk_count += 1
if sim_ltv > 80: risk_count += 1
if sim_cltv > 80: risk_count += 1
sim_band = "High Risk Indicators" if risk_count >= 3 else ("Moderate Risk Indicators" if risk_count >= 1 else "Lower Risk Indicators")
observed = df.groupby("combined_risk_indicator")["ever_delinquent_flag"].mean() * 100
estimated_rate = observed.get(sim_band, np.nan)

r1, r2 = st.columns(2)
with r1:
    st.metric("Screening result", sim_band)
with r2:
    st.metric("Observed reference delinquency rate", "—" if pd.isna(estimated_rate) else f"{estimated_rate:.2f}%")
st.caption("The reference rate is historical descriptive evidence from this sample, not a guaranteed prediction for an individual borrower.")

st.markdown('<div class="section">Filtered records</div>', unsafe_allow_html=True)
show_cols = [c for c in ["loan_sequence_number","property_state","loan_purpose","original_upb",
                          "credit_score","original_dti","original_ltv","combined_risk_indicator",
                          "ever_delinquent_flag","ever_seriously_delinquent_flag"] if c in filtered.columns]
st.dataframe(filtered[show_cols].head(1000), use_container_width=True, height=300)
st.download_button("Download filtered records", filtered.to_csv(index=False), "filtered_loan_portfolio.csv", "text/csv")

st.caption("Data source: Freddie Mac sample 2020 dataset. Delinquency and risk labels are analytical transformations documented in the project.")
