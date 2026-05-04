# Analyst CFO Main Extract

Run folder: results/res_analyst_cfo

This folder collects only the main 5/10/40 value-weighted results and the HB comparison diagnostics. No sector-neutral, exchange, size-split, equal-weighted, or other robustness tables are included.

Key files:

- `main_ff5_mom_portfolio_summary.csv`: thesis-ready main FF5+MOM raw and risk-adjusted results for LongShort and Q5.
- `main_ff5_mom_alpha_levels.csv`: annualized FF5+MOM alpha levels.
- `main_ff5_mom_alpha_differences_vs_observed.csv`: annualized FF5+MOM alpha differences versus Observed Quality.
- `main_raw_performance_all_methods.csv`: raw performance for Q5 and LongShort.
- `main_risk_adjusted_table_preview_all_models.csv`: wide preview across all factor models.
- `hb_comparison_key_summary.csv`: compact no-lead matched baseline vs analyst-CFO hybrid summary.
- `hb_no_lead_matched_vs_analyst_hybrid_by_year.csv`: yearly HB model comparison.
- `hb_sample_diagnostics_by_year.csv`: sample-size diagnostics for analyst-CFO and matched no-lead baseline.

The analyst-CFO HB model uses hybrid CFO handling: realized CFO_t+1 in training rows and analyst forecast CFO_t+1 in portfolio-year rows. The no-lead baseline comparison model uses the same estimation-window rows as the analyst-CFO hybrid model.
