# Power BI DAX Reference

Create these measures in the `final_loan_portfolio` table after importing the main CSV.

```DAX
Total Loans = COUNTROWS('final_loan_portfolio')

Total Portfolio Value = SUM('final_loan_portfolio'[original_upb])

Delinquent Loans =
SUM('final_loan_portfolio'[ever_delinquent_flag])

Seriously Delinquent Loans =
SUM('final_loan_portfolio'[ever_seriously_delinquent_flag])

Delinquency Rate =
DIVIDE([Delinquent Loans], [Total Loans], 0)

Serious Delinquency Rate =
DIVIDE([Seriously Delinquent Loans], [Total Loans], 0)

Average Credit Score =
AVERAGE('final_loan_portfolio'[credit_score])

Average DTI =
AVERAGE('final_loan_portfolio'[original_dti])

Average LTV =
AVERAGE('final_loan_portfolio'[original_ltv])
```

Format:
- Currency measures: USD
- Rate measures: Percentage with 2 decimals
- Count measures: Whole number
- Use the supplied `powerbi_theme.json` through View > Themes > Browse for themes.
