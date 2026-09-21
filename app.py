import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Loan Portfolio Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme and styling
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

st.sidebar.markdown("## ◈ Portfolio Controls")
theme = st.sidebar.radio("Appearance", ["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1)
st.session_state.theme = theme

dark = theme == "Dark"
bg = "#080d18" if dark else "#f4f7fb"
panel = "#101b30" if dark else "#ffffff"
panel_2 = "#0d1424" if dark else "#edf2f8"
text = "#edf4ff" if dark else "#172033"
muted = "#93a7c5" if dark else "#607089"
border = "#263752" if dark else "#d9e2ef"
accent = "#6ee7d0"
accent_2 = "#8b9cff"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 8% 5%, rgba(99,102,241,0.16), transparent 28%),
            radial-gradient(circle at 90% 8%, rgba(45,212,191,0.12), transparent 24%),
            {bg};
        color: {text};
    }}

    section[data-testid="stSidebar"] {{
        background: {panel_2};
        border-right: 1px solid {border};
    }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1550px;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 30px 34px;
        border: 1px solid {border};
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(99,102,241,0.28), rgba(45,212,191,0.08)),
            {panel};
        box-shadow: 0 18px 60px rgba(0,0,0,0.12);
        margin-bottom: 20px;
    }}

    .hero:after {{
        content: "◈";
        position: absolute;
        right: 35px;
        top: 10px;
        font-size: 125px;
        color: {accent};
        opacity: 0.10;
    }}

    .eyebrow {{
        color: {accent};
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .hero-title {{
        font-size: 42px;
        line-height: 1.1;
        font-weight: 900;
        margin: 8px 0;
        color: {text};
    }}

    .hero-sub {{
        color: {muted};
        font-size: 15px;
        max-width: 850px;
    }}

    .kpi {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 18px 20px;
        min-height: 122px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.06);
    }}

    .kpi-label {{
        color: {muted};
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .kpi-value {{
        color: {text};
        font-size: 29px;
        font-weight: 900;
        margin-top: 10px;
    }}

    .kpi-note {{
        color: {accent};
        font-size: 11px;
        margin-top: 5px;
    }}

    .section {{
        font-size: 21px;
        font-weight: 850;
        margin: 28px 0 10px;
        color: {text};
    }}

    .info-card {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 20px 22px;
        margin: 8px 0 14px;
        box-shadow: 0 8px 26px rgba(0,0,0,0.05);
    }}

    .info-card h4 {{
        margin-top: 0;
        color: {text};
    }}

    .info-card p, .info-card li {{
        color: {muted};
        line-height: 1.65;
    }}

    .tag {{
        display: inline-block;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 999px;
        background: rgba(110,231,208,0.12);
        color: {accent};
        border: 1px solid rgba(110,231,208,0.30);
        font-size: 11px;
        font-weight: 700;
    }}

    div[data-testid="stMetric"] {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 14px;
    }}

    .footer {{
        text-align: center;
        color: {muted};
        font-size: 12px;
        padding-top: 25px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    candidates = [
        Path("final_loan_portfolio.csv"),
        Path("data/final_loan_portfolio.csv"),
        Path("/content/final_loan_portfolio.csv"),
        Path("/mnt/data/final_loan_portfolio.csv"),
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 100:
            data = pd.read_csv(path)
            break
    else:
        return None

    numeric_cols = [
        "original_upb", "original_interest_rate", "credit_score",
        "original_dti", "original_ltv", "original_cltv",
        "delinquency_rate", "serious_delinquency_rate",
        "ever_delinquent_flag", "ever_seriously_delinquent_flag",
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["ever_delinquent_flag", "ever_seriously_delinquent_flag"]:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    return data


df = load_data()

if df is None:
    st.error("Dataset not found. Upload the CSV below or place it in the repository root/data folder.")
    uploaded = st.file_uploader("Upload final_loan_portfolio.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.caption("Filters update all charts and KPI cards.")

states = sorted(df["property_state"].dropna().astype(str).unique())
risk_values = sorted(df["combined_risk_indicator"].dropna().astype(str).unique())
purpose_values = sorted(df["loan_purpose"].dropna().astype(str).unique())

selected_states = st.sidebar.multiselect("Property state", states)
selected_risks = st.sidebar.multiselect("Analytical risk band", risk_values)
selected_purposes = st.sidebar.multiselect("Loan purpose code", purpose_values)

valid_scores = df["credit_score"].dropna()
score_low = int(max(600, valid_scores.min())) if len(valid_scores) else 600
score_high = int(min(850, valid_scores.max())) if len(valid_scores) else 850
score_range = st.sidebar.slider("Credit score range", 600, 850, (score_low, score_high))
dti_max = st.sidebar.slider("Maximum DTI (%)", 0, 65, 65)
ltv_max = st.sidebar.slider("Maximum LTV (%)", 0, 100, 100)

filtered = df.copy()

if selected_states:
    filtered = filtered[filtered["property_state"].astype(str).isin(selected_states)]
if selected_risks:
    filtered = filtered[filtered["combined_risk_indicator"].astype(str).isin(selected_risks)]
if selected_purposes:
    filtered = filtered[filtered["loan_purpose"].astype(str).isin(selected_purposes)]

filtered = filtered[
    filtered["credit_score"].between(score_range[0], score_range[1]) |
    filtered["credit_score"].isna()
]
filtered = filtered[filtered["original_dti"].fillna(0) <= dti_max]
filtered = filtered[filtered["original_ltv"].fillna(0) <= ltv_max]

# -----------------------------
# Helpers
# -----------------------------
def money(value):
    if pd.isna(value):
        return "—"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.0f}"


def plot_layout(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=58, b=10),
        font=dict(family="Inter"),
    )
    return fig


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Portfolio intelligence platform</div>
        <div class="hero-title">Loan Portfolio Intelligence</div>
        <div class="hero-sub">
            Explore portfolio exposure, delinquency patterns and analytical risk indicators
            through an interactive executive dashboard built with Python, SQL concepts and Plotly.
        </div>
        <div style="margin-top:14px;">
            <span class="tag">Python</span>
            <span class="tag">SQL / DuckDB</span>
            <span class="tag">Power BI Ready</span>
            <span class="tag">Interactive Analytics</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_dashboard, tab_risk, tab_method = st.tabs(
    ["📊 Executive Dashboard", "🧠 Risk Lab", "📘 About & Methodology"]
)

# -----------------------------
# Executive dashboard
# -----------------------------
with tab_dashboard:
    total_loans = len(filtered)
    portfolio = filtered["original_upb"].sum()
    delinquent = filtered["ever_delinquent_flag"].sum()
    serious = filtered["ever_seriously_delinquent_flag"].sum()
    delinquency_rate = delinquent / total_loans * 100 if total_loans else 0
    serious_rate = serious / total_loans * 100 if total_loans else 0
    avg_credit = filtered["credit_score"].mean() if total_loans else np.nan

    kpis = [
        ("Total loans", f"{total_loans:,.0f}", "Filtered records"),
        ("Portfolio value", money(portfolio), "Original UPB • USD"),
        ("Delinquency rate", f"{delinquency_rate:.2f}%", "Ever-delinquent proxy"),
        ("Serious delinquency", f"{serious_rate:.2f}%", "Status ≥ 3 proxy"),
        ("Average credit score", f"{avg_credit:.1f}" if not pd.isna(avg_credit) else "—", "Available scores"),
    ]

    cols = st.columns(5)
    for col, (label, value, note) in zip(cols, kpis):
        col.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">Portfolio overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        state = (
            filtered.groupby("property_state", as_index=False)
            .agg(
                total_loans=("loan_sequence_number", "count"),
                portfolio_value=("original_upb", "sum"),
                delinquency_rate=("ever_delinquent_flag", "mean"),
            )
            .sort_values("total_loans", ascending=False)
            .head(15)
        )
        state["delinquency_rate"] *= 100
        fig = px.bar(
            state,
            x="total_loans",
            y="property_state",
            orientation="h",
            text="total_loans",
            title="Top states by loan volume",
            template="plotly_dark" if dark else "plotly_white",
        )
        fig.update_layout(yaxis_title="", xaxis_title="Loans")
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    with c2:
        purpose = (
            filtered.groupby("loan_purpose", as_index=False)
            .agg(
                total_loans=("loan_sequence_number", "count"),
                delinquency_rate=("ever_delinquent_flag", "mean"),
            )
        )
        purpose["delinquency_rate"] *= 100
        fig = px.bar(
            purpose,
            x="loan_purpose",
            y="total_loans",
            color="delinquency_rate",
            title="Loan purpose volume and delinquency intensity",
            hover_data={"delinquency_rate": ":.2f"},
            template="plotly_dark" if dark else "plotly_white",
        )
        fig.update_layout(xaxis_title="Purpose code", yaxis_title="Loans")
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    st.markdown('<div class="section">Risk intelligence</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        risk = (
            filtered.groupby("combined_risk_indicator", as_index=False)
            .agg(
                total_loans=("loan_sequence_number", "count"),
                delinquency_rate=("ever_delinquent_flag", "mean"),
                serious_rate=("ever_seriously_delinquent_flag", "mean"),
            )
        )
        risk["delinquency_rate"] *= 100
        risk["serious_rate"] *= 100
        fig = px.bar(
            risk,
            x="combined_risk_indicator",
            y="delinquency_rate",
            color="combined_risk_indicator",
            title="Delinquency rate by analytical risk band",
            hover_data={"total_loans": True, "serious_rate": ":.2f"},
            template="plotly_dark" if dark else "plotly_white",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Delinquency rate (%)", showlegend=False)
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    with c4:
        sample = filtered.sample(min(7000, len(filtered)), random_state=42) if len(filtered) else filtered
        if len(sample):
            fig = px.scatter(
                sample,
                x="original_ltv",
                y="original_dti",
                color="credit_risk_category",
                size="original_upb",
                hover_data=["property_state", "loan_purpose"],
                title="Exposure map: LTV versus DTI",
                template="plotly_dark" if dark else "plotly_white",
            )
            fig.update_layout(xaxis_title="Original LTV (%)", yaxis_title="Original DTI (%)")
            st.plotly_chart(plot_layout(fig), use_container_width=True)
        else:
            st.info("No records match the selected filters.")

    st.markdown('<div class="section">Portfolio composition</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        credit = (
            filtered["credit_risk_category"]
            .value_counts(dropna=False)
            .rename_axis("category")
            .reset_index(name="loans")
        )
        fig = px.pie(
            credit,
            names="category",
            values="loans",
            hole=0.55,
            title="Credit risk mix",
            template="plotly_dark" if dark else "plotly_white",
        )
        st.plotly_chart(plot_layout(fig, 390), use_container_width=True)

    with c6:
        dti = (
            filtered["dti_risk_category"]
            .value_counts(dropna=False)
            .rename_axis("category")
            .reset_index(name="loans")
        )
        fig = px.bar(
            dti,
            x="category",
            y="loans",
            color="loans",
            title="DTI risk distribution",
            template="plotly_dark" if dark else "plotly_white",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Loans")
        st.plotly_chart(plot_layout(fig, 390), use_container_width=True)

    st.markdown('<div class="section">Filtered records</div>', unsafe_allow_html=True)
    show_cols = [
        c for c in [
            "loan_sequence_number", "property_state", "loan_purpose",
            "original_upb", "credit_score", "original_dti", "original_ltv",
            "combined_risk_indicator", "ever_delinquent_flag",
            "ever_seriously_delinquent_flag"
        ] if c in filtered.columns
    ]
    st.dataframe(filtered[show_cols].head(1000), use_container_width=True, height=300)
    st.download_button(
        "⬇ Download filtered records",
        filtered.to_csv(index=False),
        "filtered_loan_portfolio.csv",
        "text/csv",
    )

# -----------------------------
# Risk lab
# -----------------------------
with tab_risk:
    st.markdown('<div class="section">Transparent risk scenario simulator</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h4>How this section works</h4>
            <p>
            Enter a hypothetical borrower profile. The tool applies transparent analytical
            thresholds to classify the profile into a risk-indicator band. It is designed
            for learning and portfolio screening—not for approved lending decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        sim_score = st.number_input("Credit score", 300, 850, 720)
    with s2:
        sim_dti = st.number_input("DTI (%)", 0.0, 100.0, 35.0)
    with s3:
        sim_ltv = st.number_input("LTV (%)", 0.0, 150.0, 80.0)
    with s4:
        sim_cltv = st.number_input("CLTV (%)", 0.0, 150.0, 80.0)

    risk_count = 0
    if sim_score < 670:
        risk_count += 1
    if sim_dti > 35:
        risk_count += 1
    if sim_ltv > 80:
        risk_count += 1
    if sim_cltv > 80:
        risk_count += 1

    sim_band = (
        "High Risk Indicators" if risk_count >= 3
        else "Moderate Risk Indicators" if risk_count >= 1
        else "Lower Risk Indicators"
    )

    observed = df.groupby("combined_risk_indicator")["ever_delinquent_flag"].mean() * 100
    reference_rate = observed.get(sim_band, np.nan)

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Screening result", sim_band)
    with r2:
        st.metric(
            "Observed reference delinquency",
            "—" if pd.isna(reference_rate) else f"{reference_rate:.2f}%"
        )

    st.info(
        "The reference rate is descriptive evidence from this sample. "
        "It is not an individual-level prediction or a validated underwriting score."
    )

    st.markdown('<div class="section">Indicator logic</div>', unsafe_allow_html=True)
    logic = pd.DataFrame({
        "Indicator": ["Credit score", "DTI", "LTV", "CLTV"],
        "Analytical trigger": ["Below 670", "Above 35%", "Above 80%", "Above 80%"],
        "Meaning": [
            "Lower credit-score segment",
            "Higher debt burden relative to income",
            "Higher loan exposure relative to property value",
            "Higher combined loan exposure relative to property value",
        ],
    })
    st.dataframe(logic, use_container_width=True, hide_index=True)

# -----------------------------
# About and methodology
# -----------------------------
with tab_method:
    st.markdown('<div class="section">What problem does this project solve?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h4>Business problem</h4>
            <p>
            Loan portfolios contain thousands of records with information about borrower
            credit characteristics, loan exposure, geography and repayment performance.
            Reviewing these records manually makes it difficult to identify concentration,
            delinquency patterns and segments that require closer monitoring.
            </p>
            <h4>Solution delivered</h4>
            <p>
            This project converts raw loan-level data into an interactive analytics product.
            It combines data cleaning, feature engineering, SQL-style aggregation and
            dashboard visualisation so users can explore portfolio exposure and risk
            patterns through filters and charts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section">Dataset overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h4>Source</h4>
            <p>Freddie Mac Single-Family Loan-Level Dataset sample for 2020.</p>
            <h4>Key data elements</h4>
            <ul>
                <li>Loan amount / original unpaid principal balance (UPB)</li>
                <li>Credit score, DTI, LTV and CLTV</li>
                <li>Property state and loan purpose</li>
                <li>Interest rate, loan term and borrower-related fields</li>
                <li>Monthly performance and delinquency status transformations</li>
            </ul>
            <h4>Unit of money</h4>
            <p>Loan amounts are represented in US dollars (USD), not Indian rupees.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section">Model and analytical methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h4>What model is used?</h4>
            <p>
            The current application uses descriptive analytics and an explainable,
            rule-based scenario simulator. It is not a trained machine-learning default
            prediction model and should not be presented as a production credit-scoring system.
            </p>
            <h4>Feature engineering</h4>
            <ul>
                <li>Credit score categories: Fair, Good, Very Good and Excellent</li>
                <li>DTI categories: Low, Moderate, High and Very High</li>
                <li>LTV and CLTV risk categories</li>
                <li>Combined analytical risk-indicator band</li>
                <li>Ever-delinquent and serious-delinquency proxy flags</li>
            </ul>
            <h4>Why this approach?</h4>
            <p>
            The rules are transparent and easy to explain during a portfolio review,
            viva, interview or business presentation. A production model would require
            a clearly defined target, time-based validation, model monitoring,
            bias testing and approval controls.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section">Important interpretation notes</div>', unsafe_allow_html=True)
    st.warning(
        "Delinquency labels are analytical transformations of performance data and are not "
        "official NPA classifications. The dataset does not directly provide a branch field; "
        "property state is used as a geographic proxy. Any future profitability calculation "
        "would require explicit assumptions because lender revenue and costs are not directly provided."
    )

    st.markdown('<div class="section">Technology architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <p><b>Data source → Python cleaning → Feature engineering → SQL analysis → Power BI reporting → Streamlit application</b></p>
            <p>
            Python prepares the dataset, SQL supports aggregation and analysis,
            Power BI can be used for business reporting, and Streamlit provides
            an interactive web interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">Loan Portfolio Intelligence • Educational analytics product • Data source: Freddie Mac sample 2020</div>',
    unsafe_allow_html=True,
)
