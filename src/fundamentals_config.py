SHEETS = {
    "valuation": "Valuation",
    "income_statement": "Income Statement",
    "balance_sheet": "Balance Sheet",
    "cash_flow": "Cash Flow",
}

FIELDS = {
    # --- Income statement  ---
    "revenue": "Revenue from Business Activities - Total",
    "gross_profit": "Gross Profit - Industrials/Property - Total",
    "ebit": "Earnings before Interest & Taxes (EBIT)",
    "ebitda": "Earnings before Interest, Taxes, Depreciation & Amortization (EBITDA)",
    "net_income_before_minority": "Net Income before Minority Interest",
    "net_income": "Net Income after Minority Interest",

    # --- Balance sheet ---
    "total_assets": "Total Assets",
    "total_equity": "Common Equity - Total",
    "cash": "Cash & Short Term Investments - Total",
    "current_assets": "Total Current Assets",
    "current_liabilities": "Total Current Liabilities",
    "short_term_debt": "Short-Term Debt & Current Portion of Long-Term Debt",
    "long_term_debt": "Debt - Long-Term - Total",

    # --- Cash flow ---
    "ocf": "Net Cash Flow from Operating Activities",
    "capex": "Capital Expenditures - Total",
    "dividends_paid": "Dividends Paid - Cash - Total - Cash Flow",
    "da_cf_reconcile": "Depreciation, Depletion & Amortization including Impairment - Cash Flow - to Reconcile",
    "wc_change_cf": "Working Capital - Increase/(Decrease) - Cash Flow",

    # --- Financial Summary ---
    "market_cap": "Market Capitalization",
    "shares_outstanding": "Common Shares - Outstanding - Total",
    "shares_diluted": "Shares used to calculate Diluted EPS - Total",
}

METRICS = {
    # Income statement
    "revenue": {"statement": "income_statement", "field": FIELDS["revenue"]},
    "gross_profit": {"statement": "income_statement", "field": FIELDS["gross_profit"]},
    "ebit": {"statement": "income_statement", "field": FIELDS["ebit"]},
    "ebitda": {"statement": "income_statement", "field": FIELDS["ebitda"]},
    "net_income_before_minority": {"statement": "income_statement", "field": FIELDS["net_income_before_minority"]},
    "net_income": {"statement": "income_statement", "field": FIELDS["net_income"]},
    "shares_diluted": {"statement": "income_statement", "field": FIELDS["shares_diluted"]},

    # Balance sheet
    "total_assets": {"statement": "balance_sheet", "field": FIELDS["total_assets"]},
    "total_equity": {"statement": "balance_sheet", "field": FIELDS["total_equity"]},
    "cash": {"statement": "balance_sheet", "field": FIELDS["cash"]},
    "current_assets": {"statement": "balance_sheet", "field": FIELDS["current_assets"]},
    "current_liabilities": {"statement": "balance_sheet", "field": FIELDS["current_liabilities"]},
    "short_term_debt": {"statement": "balance_sheet", "field": FIELDS["short_term_debt"]},
    "long_term_debt": {"statement": "balance_sheet", "field": FIELDS["long_term_debt"]},
    "shares_outstanding": {"statement": "balance_sheet", "field": FIELDS["shares_outstanding"]},

    # Cash flow
    "ocf": {"statement": "cash_flow", "field": FIELDS["ocf"]},
    "capex": {"statement": "cash_flow", "field": FIELDS["capex"]},
    "dividends_paid": {"statement": "cash_flow", "field": FIELDS["dividends_paid"]},
    "da_cf_reconcile": {"statement": "cash_flow", "field": FIELDS["da_cf_reconcile"]},
    "wc_change_cf": {"statement": "cash_flow", "field": FIELDS["wc_change_cf"]},

    # Valuation
    "market_cap": {"statement": "valuation", "field": FIELDS["market_cap"]},
}
