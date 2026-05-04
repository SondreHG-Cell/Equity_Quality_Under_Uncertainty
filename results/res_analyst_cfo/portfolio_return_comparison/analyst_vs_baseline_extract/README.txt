Analyst CFO vs Baseline HB Portfolio Return Comparison

This folder compares portfolio returns for two matched-sample HB specifications.

Specifications:
- analyst_cfo_hybrid_point: analyst-CFO hybrid HB, where training rows use realized CFO_t+1 and portfolio-year rows use analyst forecast CFO_t+1.
- no_cfo_lead_matched_point: baseline HB without CFO_t+1, restricted to the same matched analyst-CFO firm-year sample.

Important note: the uploaded result folder does not contain sigma_posteriors_full.parquet, so this comparison uses point EB latent-quality construction for both specifications. That keeps the analyst and baseline return comparison internally comparable, but it is not the same as the uploaded analyst-only full-propagation downstream run.

Key files:
- portfolio_return_comparison_ff5_mom.csv
- analyst_minus_baseline_ff5_mom_deltas.csv
- raw_performance_both_specs.csv
- alpha_levels_all_models_both_specs.csv
- alpha_differences_all_models_both_specs.csv
- risk_adjusted_preview_all_models_both_specs.csv
- hb_model_comparison_by_year.csv
- hb_model_comparison_overall.csv
- hb_sample_diagnostics_by_year.csv
