Analyst CFO vs no-CFO-lead matched baseline portfolio comparison

This folder compares two point-estimate portfolio pipelines on the matched analyst-CFO sample:

1. Analyst CFO hybrid HB
   Source tables: /Users/simenoiseth/Desktop/Equity_Quality_Under_Uncertainty/results/res_analyst_cfo/portfolio_return_comparison/analyst_cfo_hybrid_point/portfolio_evaluation/thesis_risk_adjusted_tables_ucits_5_10_40
   Formation-year CFO_t+1 input uses analyst CFO forecasts; historical training observations can use realized CFO_t+1 once observable.

2. Baseline no-lead HB matched sample
   Source tables: /Users/simenoiseth/Desktop/Equity_Quality_Under_Uncertainty/results/res_analyst_cfo/portfolio_return_comparison/no_cfo_lead_matched_point/portfolio_evaluation/thesis_risk_adjusted_tables_ucits_5_10_40
   Baseline HB omits CFO_t+1 and is restricted to the same analyst-forecast matched firm-year window.

Main files:
- portfolio_return_comparison_ff5_mom.csv: FF5+MOM alpha levels plus raw performance for both specs.
- analyst_minus_baseline_ff5_mom_deltas.csv: analyst-minus-baseline deltas by method and strategy.
- headline_ff5_mom_deltas.csv: compact thesis-facing deltas.
- raw_performance_both_specs.csv: unadjusted Q5 and long-short performance.
- alpha_levels_all_models_both_specs.csv: all alpha-level regressions.
- alpha_differences_all_models_both_specs.csv: all alpha-difference tests.
- risk_adjusted_preview_all_models_both_specs.csv: wide preview from each generator.
- monthly_portfolio_returns_used_both_specs.csv: monthly Q5 and long-short returns used.
- portfolio_assignment_changes.csv: firm-year assignment changes by method.
- hb_model_comparison_by_year.csv and hb_model_comparison_overall.csv: upstream HB expected-accrual comparison diagnostics.
