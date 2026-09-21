import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Loan Portfolio Intelligence | V2", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# ------------------------- Theme -------------------------
theme = st.sidebar.radio("Interface theme", ["Dark", "Light"], horizontal=True)
dark = theme == "Dark"
bg = "#08111f" if dark else "#f5f7fb"
card = "#101e33" if dark else "#ffffff"
text = "#edf4ff" if dark else "#172033"
muted = "#9aacc8" if dark else "#65718a"
border = "#2a3b58" if dark else "#dbe2ef"
plot_template = "plotly_dark" if dark else "plotly_white"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: Inter, sans-serif; }}
.stApp {{ background: {bg}; color: {text}; }}
section[data-testid="stSidebar"] {{ background: {card}; border-right: 1px solid {border}; }}
.block-container {{ max-width: 1550px; padding-top: 1.5rem; }}
h1,h2,h3,h4 {{ color: {text} !important; letter-spacing: -.5px; }}
.hero {{ padding: 26px 30px; border: 1px solid {border}; border-radius: 22px; background: linear-gradient(135deg, #12345b, #15203b 55%, #1e2850); margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.12); }}
.hero-title {{ font-size: 35px; font-weight: 800; color: #ffffff; line-height: 1.1; }}
.hero-sub {{ color: #c4d5ef; margin-top: 9px; font-size: 14px; }}
.badge {{ display:inline-block; padding:5px 10px; border-radius:30px; background:#1dd6b0; color:#06251f; font-size:11px; font-weight:800; margin-top:13px; }}
.kpi {{ background: {card}; border: 1px solid {border}; border-radius: 17px; padding: 17px 19px; min-height: 118px; box-shadow: 0 5px 18px rgba(0,0,0,.05); }}
.kpi-label {{ color: {muted}; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}
.kpi-value {{ color: {text}; font-size: 27px; font-weight: 800; margin-top: 10px; }}
.kpi-note {{ color: #16b99b; font-size: 11px; margin-top: 5px; }}
.panel {{ background:{card}; border:1px solid {border}; border-radius:16px; padding:18px; margin:10px 0; }}
.note {{ color:{muted}; font-size:12px; }}
div[data-testid="stMetric"] {{ background:{card}; border:1px solid {border}; border-radius:14px; padding:12px; }}
button {{ border-radius:10px !important; }}
</style>
""", unsafe_allow_html=True)

# ------------------------- Data -------------------------
@st.cache_data(show_spinner="Loading the Freddie Mac sample...")
def load_data():
    paths = [Path("final_loan_portfolio.csv"), Path("data/final_loan_portfolio.csv"), Path("/content/final_loan_portfolio.csv"), Path("/mnt/data/final_loan_portfolio.csv")]
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        return None
    numeric = ["original_upb", "original_interest_rate", "credit_score", "original_dti", "original_ltv", "original_cltv", "original_loan_term", "number_of_borrowers", "ever_delinquent_flag", "ever_seriously_delinquent_flag"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["ever_delinquent_flag", "ever_seriously_delinquent_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df

def money(value):
    if pd.isna(value): return "—"
    value = float(value)
    if abs(value) >= 1e9: return f"${value/1e9:.2f}B"
    if abs(value) >= 1e6: return f"${value/1e6:.1f}M"
    return f"${value:,.0f}"

def pct(value):
    return "—" if pd.isna(value) else f"{value:.2f}%"

def safe_fig(fig, height=390):
    fig.update_layout(template=plot_template, height=height, margin=dict(l=10,r=10,t=58,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

df = load_data()
if df is None:
    st.error("Dataset not found. Keep final_loan_portfolio.csv in the repository root or data/ folder.")
    st.stop()

# ------------------------- Header -------------------------
st.markdown("""
<div class="hero">
  <div class="hero-title">Loan Portfolio Intelligence</div>
  <div class="hero-sub">Executive portfolio analytics · Risk segmentation · Explainable ML · Scenario planning</div>
  <span class="badge">REAL HISTORICAL DATA · FREDDIE MAC 2020 SAMPLE</span>
</div>
""", unsafe_allow_html=True)

# ------------------------- Sidebar -------------------------
st.sidebar.markdown("## ◈ Portfolio Controls")
st.sidebar.caption("All analytical pages respond to these filters.")

def values(col):
    return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else []

selected_states = st.sidebar.multiselect("Property state", values("property_state"))
selected_risks = st.sidebar.multiselect("Analytical risk band", values("combined_risk_indicator"))
selected_purposes = st.sidebar.multiselect("Loan purpose", values("loan_purpose"))

if "credit_score" in df.columns:
    valid_scores = df.credit_score.dropna()
    low_score = int(max(300, np.floor(valid_scores.min()))) if len(valid_scores) else 300
    high_score = int(min(900, np.ceil(valid_scores.max()))) if len(valid_scores) else 850
    score_range = st.sidebar.slider("Credit score range", low_score, high_score, (low_score, high_score))
else:
    score_range = (300, 900)
dti_max = st.sidebar.slider("Maximum DTI (%)", 0, 100, 65)
ltv_max = st.sidebar.slider("Maximum LTV (%)", 0, 150, 100)

filtered = df.copy()
if selected_states and "property_state" in filtered: filtered = filtered[filtered.property_state.astype(str).isin(selected_states)]
if selected_risks and "combined_risk_indicator" in filtered: filtered = filtered[filtered.combined_risk_indicator.astype(str).isin(selected_risks)]
if selected_purposes and "loan_purpose" in filtered: filtered = filtered[filtered.loan_purpose.astype(str).isin(selected_purposes)]
if "credit_score" in filtered: filtered = filtered[filtered.credit_score.isna() | filtered.credit_score.between(score_range[0], score_range[1])]
if "original_dti" in filtered: filtered = filtered[filtered.original_dti.isna() | (filtered.original_dti <= dti_max)]
if "original_ltv" in filtered: filtered = filtered[filtered.original_ltv.isna() | (filtered.original_ltv <= ltv_max)]

n = len(filtered)
portfolio = filtered.original_upb.sum() if "original_upb" in filtered else 0
bad_rate = filtered.ever_delinquent_flag.mean()*100 if n else 0
serious_rate = filtered.ever_seriously_delinquent_flag.mean()*100 if n else 0
avg_credit = filtered.credit_score.mean() if n else np.nan

kpis = [("Total loans", f"{n:,}", "Filtered records"), ("Portfolio value", money(portfolio), "Original UPB · USD"), ("Ever delinquent", pct(bad_rate), "Historical proxy"), ("Serious delinquency", pct(serious_rate), "Historical proxy"), ("Average credit", f"{avg_credit:.1f}" if not pd.isna(avg_credit) else "—", "Available scores")]
cols = st.columns(5)
for col, (label, value, note) in zip(cols, kpis):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

# ------------------------- Pages -------------------------
tabs = st.tabs(["Executive Dashboard", "Risk Intelligence", "ML Classification", "Scenario Lab", "Problem & Methodology", "Data Explorer"])

with tabs[0]:
    st.subheader("Executive overview")
    if n:
        top_state = filtered.property_state.value_counts().index[0] if "property_state" in filtered else "N/A"
        high = (filtered.combined_risk_indicator.astype(str).str.contains("High", na=False).sum()) if "combined_risk_indicator" in filtered else 0
        st.info(f"**Portfolio signal:** {top_state} has the highest filtered loan count. **Risk signal:** {high:,} records fall in the project-defined high-risk indicator band. These are descriptive signals, not lending decisions.")
    a, b = st.columns(2)
    with a:
        if "property_state" in filtered.columns:
            state = filtered.groupby("property_state", as_index=False).agg(loans=("property_state", "size"), value=("original_upb", "sum")).sort_values("loans", ascending=False).head(15)
            st.plotly_chart(safe_fig(px.bar(state, x="loans", y="property_state", orientation="h", text="loans", title="Loan volume by property state"), 420), use_container_width=True)
    with b:
        if "loan_purpose" in filtered.columns:
            purpose = filtered.groupby("loan_purpose", as_index=False).agg(loans=("loan_purpose", "size"), delinquency=("ever_delinquent_flag", "mean"))
            purpose["delinquency"] *= 100
            st.plotly_chart(safe_fig(px.bar(purpose, x="loan_purpose", y="loans", color="delinquency", title="Loan purpose volume and delinquency intensity", hover_data={"delinquency":":.2f"}), 420), use_container_width=True)
    a, b = st.columns(2)
    with a:
        if "combined_risk_indicator" in filtered.columns:
            risk = filtered.groupby("combined_risk_indicator", as_index=False).agg(loans=("combined_risk_indicator", "size"), delinquency=("ever_delinquent_flag", "mean"))
            risk["delinquency"] *= 100
            st.plotly_chart(safe_fig(px.bar(risk, x="combined_risk_indicator", y="delinquency", color="combined_risk_indicator", title="Observed delinquency by analytical risk band"), 400), use_container_width=True)
    with b:
        if {"original_ltv", "original_dti", "credit_risk_category"}.issubset(filtered.columns):
            sample = filtered.sample(min(6000, len(filtered)), random_state=42) if len(filtered) else filtered
            st.plotly_chart(safe_fig(px.scatter(sample, x="original_ltv", y="original_dti", color="credit_risk_category", size="original_upb", hover_data=[c for c in ["property_state", "loan_purpose"] if c in sample.columns], title="Exposure map: LTV vs DTI"), 400), use_container_width=True)
    st.markdown("#### Recommended business investigation areas")
    st.markdown("- Review geographic concentration before expanding comparable exposure.\n- Segment monitoring by DTI, LTV, credit-score and analytical risk bands.\n- Use the ML output as a research screening aid, not a validated underwriting engine.")

with tabs[1]:
    st.subheader("Risk intelligence")
    st.caption("Risk labels and bands are project-defined analytical transformations.")
    a, b = st.columns(2)
    with a:
        if {"credit_score", "original_dti", "ever_delinquent_flag"}.issubset(filtered.columns):
            m = filtered.copy()
            m["credit_band"] = pd.cut(m.credit_score, [0, 669, 739, 799, 1000], labels=["Below 670", "670–739", "740–799", "800+"])
            m["dti_band"] = pd.cut(m.original_dti, [-1, 20, 35, 45, 1000], labels=["Low ≤20", "Moderate 21–35", "High 36–45", "Very high >45"])
            heat = m.pivot_table(index="dti_band", columns="credit_band", values="ever_delinquent_flag", aggfunc="mean", observed=False)*100
            st.markdown("**Credit score × DTI delinquency matrix (%)**")
            st.dataframe(heat.round(2), use_container_width=True)
            st.caption("Blank cells indicate no usable observations in that segment.")
    with b:
        if "combined_risk_indicator" in filtered.columns:
            counts = filtered.combined_risk_indicator.value_counts(dropna=False).rename_axis("risk_band").reset_index(name="loans")
            st.plotly_chart(safe_fig(px.pie(counts, names="risk_band", values="loans", hole=.55, title="Portfolio mix by analytical risk band"), 380), use_container_width=True)
    if {"dti_risk_category", "ever_delinquent_flag"}.issubset(filtered.columns):
        dti = filtered.groupby("dti_risk_category", as_index=False).agg(loans=("dti_risk_category", "size"), delinquency=("ever_delinquent_flag", "mean"))
        dti["delinquency"] *= 100
        st.plotly_chart(safe_fig(px.bar(dti, x="dti_risk_category", y="delinquency", color="dti_risk_category", title="Observed delinquency by DTI risk category"), 390), use_container_width=True)

with tabs[2]:
    st.subheader("Explainable ML classification")
    st.caption("Logistic Regression predicts the historical ever-delinquent flag using origination attributes. It is a research demonstration, not production underwriting validation.")
    features = [c for c in ["credit_score", "original_dti", "original_ltv", "original_cltv", "original_interest_rate", "original_upb", "original_loan_term", "number_of_borrowers", "property_state", "loan_purpose", "occupancy_status", "channel"] if c in df.columns]
    target = "ever_delinquent_flag"
    if len(features) < 2 or target not in df.columns:
        st.warning("Required model columns are missing from the dataset.")
    else:
        model_df = df[features + [target]].dropna(subset=[target]).copy()
        X, y = model_df[features], model_df[target].astype(int)
        if y.nunique() < 2:
            st.warning("The target contains only one class; model metrics cannot be calculated.")
        else:
            categorical = [c for c in features if X[c].dtype == "object"]
            numeric = [c for c in features if c not in categorical]
            pre = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
            model = Pipeline([("preprocess", pre), ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced"))])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=42)
            with st.spinner("Training model on the real dataset..."):
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                probability = model.predict_proba(X_test)[:, 1]
            metrics = [accuracy_score(y_test, prediction), precision_score(y_test, prediction, zero_division=0), recall_score(y_test, prediction, zero_division=0), f1_score(y_test, prediction, zero_division=0), roc_auc_score(y_test, probability)]
            metric_cols = st.columns(5)
            for col, label, value in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], metrics): col.metric(label, f"{value:.3f}")
            a, b = st.columns(2)
            with a:
                cm = confusion_matrix(y_test, prediction)
                st.markdown("**Confusion matrix**")
                st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]), use_container_width=True)
            with b:
                st.markdown("**Interpretation**")
                st.markdown("- Precision: share of predicted delinquent cases that were delinquent in the test sample.\n- Recall: share of actual delinquent cases detected by the model.\n- ROC-AUC: ranking ability across thresholds.\n- Metrics are sample-specific and should be validated before operational use.")

with tabs[3]:
    st.subheader("Scenario lab")
    st.caption("Illustrative stress testing using user-entered assumptions. This is not observed lender loss.")
    stress = st.slider("Additional delinquency uplift (percentage points)", 0.0, 20.0, 5.0, .5)
    recovery = st.slider("Assumed recovery rate (%)", 0, 100, 60)
    base = (filtered.ever_delinquent_flag.mean() if n else 0)
    stressed = min(base + stress/100, 1)
    loss = portfolio * stressed * (1 - recovery/100)
    a, b, c = st.columns(3)
    a.metric("Observed base rate", f"{base*100:.2f}%")
    b.metric("Stress-adjusted rate", f"{stressed*100:.2f}%")
    c.metric("Illustrative exposure loss", money(loss))
    st.warning("Stress uplift and recovery are assumptions supplied in the interface. They are not Freddie Mac loss observations.")
    st.markdown("#### Scenario interpretation")
    st.write(f"Under the selected assumptions, the model applies a {stress:.1f}-percentage-point uplift to the filtered historical delinquency rate and assumes {recovery}% recovery. The resulting amount is an illustrative exposure calculation, not a forecast.")

with tabs[4]:
    st.subheader("What problem does this project solve?")
    st.markdown("""
    <div class="panel"><b>Business problem</b><br>Loan portfolios contain many records across borrower characteristics, property locations, loan terms and performance outcomes. Reviewing these records manually makes it difficult to identify concentration, delinquency patterns and segments that deserve further investigation.</div>
    <div class="panel"><b>Solution delivered</b><br>This application converts the historical loan-level dataset into an interactive decision-support dashboard with portfolio KPIs, geographic and purpose-level analysis, risk segmentation, a transparent classification model and assumption-based stress testing.</div>
    <div class="panel"><b>Business value</b><br>Analysts can filter the portfolio, compare segments, identify descriptive risk patterns, test assumptions and download a focused data extract for further investigation.</div>
    """, unsafe_allow_html=True)
    st.markdown("#### Methodology and limitations")
    st.markdown("- Source: Freddie Mac 2020 sample data used in this project.\n- Currency: USD; `original_upb` represents original unpaid principal balance.\n- Delinquency and serious-delinquency flags are historical sample indicators.\n- The combined risk indicator is a project-defined analytical band, not an official lender grade.\n- NPA, profitability and lender losses are not directly observed in the dataset.\n- The ML model is a demonstration and has not been validated for production lending decisions.\n- The app is interactive, but the underlying 2020 dataset is historical/static; it is not a real-time market feed.")
    st.markdown("#### Technology stack")
    st.code("Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit", language="text")

with tabs[5]:
    st.subheader("Filtered data explorer")
    st.caption("Preview up to 1,000 filtered rows. Download the complete filtered extract below.")
    preferred = ["loan_sequence_number", "property_state", "loan_purpose", "original_upb", "credit_score", "original_dti", "original_ltv", "original_cltv", "combined_risk_indicator", "ever_delinquent_flag", "ever_seriously_delinquent_flag"]
    show = [c for c in preferred if c in filtered.columns]
    st.dataframe(filtered[show].head(1000), use_container_width=True, height=420)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False), "filtered_loan_portfolio.csv", "text/csv")

st.divider()
st.caption("Source: Freddie Mac 2020 sample used in this project. Risk bands are analytical transformations. Currency is USD. The dataset is historical/static; interface refresh does not create real-time loan data.")
