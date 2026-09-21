# Loan Portfolio Intelligence Dashboard

An interactive dark-theme loan portfolio analytics application built with **Python, SQL-ready data, Plotly, and Streamlit**.

## Features

- Executive KPI cards
- Interactive state, risk, purpose, credit score, DTI, and LTV filters
- State-wise portfolio concentration
- Loan purpose comparison
- Risk-band delinquency analysis
- Credit risk mix and DTI distribution
- LTV vs DTI exposure scatter plot
- Explainable rule-based risk scenario simulator
- Filtered data preview and CSV download
- Power BI-ready source dataset

## Important analytical limitations

- Freddie Mac sample data does not contain a direct branch field. `property_state` is used as a geographic proxy.
- NPA is not directly available. Delinquency and serious delinquency flags are proxies based on performance status.
- The scenario simulator is an explainable screening demonstration, not a validated lending or underwriting model.
- Loan amounts are in USD.
- Check the dataset's licensing and redistribution terms before publishing raw data publicly.

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Deploy

The app can be deployed through Streamlit Community Cloud after pushing the repository to GitHub. Keep the CSV in the `data` folder only if redistribution is permitted; otherwise use a private data source or upload it through the app.

## Power BI

Use `data/final_loan_portfolio.csv` as the primary Power BI source. The included Power BI theme and DAX reference files are provided separately in the project package.
