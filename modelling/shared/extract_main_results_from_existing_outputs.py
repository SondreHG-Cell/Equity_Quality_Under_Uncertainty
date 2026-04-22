# extract_main_results_from_existing_outputs.py

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# Project / run paths
# -----------------------------------------------------------------------------

def find_project_root() -> Path:
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
    for p in [here] + list(here.parents):
        if (p / "results").exists() & (p / "modelling").exists():
            return p
    raise FileNotFoundError("Could not find project root containing a 'results' folder.")


PROJECT_ROOT = find_project_root()
RUN_DIR = PROJECT_ROOT / "results" / "first_hb_run"
PORT_EVAL_DIR = RUN_DIR / "portfolio_evaluation"
OUTPUT_DIR = RUN_DIR / "main_results_extract"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_CONFIG_JSON = RUN_DIR / "run_config.json"
RUN_SUMMARY_JSON = RUN_DIR / "run_summary.json"

RAW_PERF_CSV = PORT_EVAL_DIR / "raw_performance.csv"
RISK_ADJ_CSV = PORT_EVAL_DIR / "risk_adjusted_performance.csv"
ALPHA_DIFF_CSV = PORT_EVAL_DIR / "alpha_differences.csv"
MONTHLY_RETURNS_CSV = PORT_EVAL_DIR / "monthly_portfolio_returns.csv"

required = [
    RUN_DIR,
    PORT_EVAL_DIR,
    RAW_PERF_CSV,
    RISK_ADJ_CSV,
    ALPHA_DIFF_CSV,
    MONTHLY_RETURNS_CSV,
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files/folders:\n" + "\n".join(missing))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def maybe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_monthly_portfolio_returns(path: Path) -> pd.DataFrame:
    # Expected save format from returns_wide.to_csv(): MultiIndex columns, date index
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()].copy()
        if isinstance(df.columns, pd.MultiIndex):
            return df.sort_index()
    except Exception:
        pass

    raise ValueError(
        f"Could not read {path} as wide monthly portfolio returns with MultiIndex columns."
    )


def find_ls_col(sub: pd.DataFrame) -> str | None:
    for candidate in ["LS", "Q5-Q1"]:
        if candidate in sub.columns:
            return candidate
    return None


# -----------------------------------------------------------------------------
# Load existing outputs
# -----------------------------------------------------------------------------

run_config = maybe_read_json(RUN_CONFIG_JSON)
run_summary = maybe_read_json(RUN_SUMMARY_JSON)

raw_perf = pd.read_csv(RAW_PERF_CSV)
risk_adj = pd.read_csv(RISK_ADJ_CSV)
alpha_diff = pd.read_csv(ALPHA_DIFF_CSV)
monthly_returns = load_monthly_portfolio_returns(MONTHLY_RETURNS_CSV)

# Standardize LS label
if "Portfolio" in raw_perf.columns:
    raw_perf["Portfolio"] = raw_perf["Portfolio"].replace({"LS": "Q5-Q1"})
if "Portfolio" in risk_adj.columns:
    risk_adj["Portfolio"] = risk_adj["Portfolio"].replace({"LS": "Q5-Q1"})

# Figure out whether LS already exists
methods = list(pd.Index(monthly_returns.columns.get_level_values(0)).unique())
ls_already_exists = True
for method in methods:
    sub = monthly_returns[method]
    if find_ls_col(sub) is None:
        ls_already_exists = False
        break


# -----------------------------------------------------------------------------
# Print identified inputs / heads
# -----------------------------------------------------------------------------

print("\n[1] Files identified as inputs:")
print(f"  run_dir                     = {RUN_DIR}")
print(f"  raw_performance_csv         = {RAW_PERF_CSV}")
print(f"  risk_adjusted_csv           = {RISK_ADJ_CSV}")
print(f"  alpha_differences_csv       = {ALPHA_DIFF_CSV}")
print(f"  monthly_portfolio_returns   = {MONTHLY_RETURNS_CSV}")

if run_config is not None and "factors_csv" in run_config:
    print(f"  factor_returns_file         = {run_config['factors_csv']}")
elif run_summary is not None and "config" in run_summary and "factors_csv" in run_summary["config"]:
    print(f"  factor_returns_file         = {run_summary['config']['factors_csv']}")
else:
    print("  factor_returns_file         = not read from disk here; already embedded in existing evaluation outputs")

print("\n[2] Portfolio returns file:")
print(f"  {MONTHLY_RETURNS_CSV}")

print("\n[3] Factor returns file:")
if run_config is not None and "factors_csv" in run_config:
    print(f"  {run_config['factors_csv']}")
elif run_summary is not None and "config" in run_summary and "factors_csv" in run_summary["config"]:
    print(f"  {run_summary['config']['factors_csv']}")
else:
    print("  not reloaded; using existing portfolio_evaluation outputs")

print("\n[4] Long-short series:")
print("  already exists in monthly_portfolio_returns.csv" if ls_already_exists else "  must be constructed from Q5 and Q1")

print("\nraw_perf.head()")
print(raw_perf.head())

print("\nrisk_adj.head()")
print(risk_adj.head())

print("\nalpha_diff.head()")
print(alpha_diff.head())

print("\nmonthly_returns.head()")
print(monthly_returns.head())


# -----------------------------------------------------------------------------
# 1) Main results table: Q5-Q1 long-short across methods
# -----------------------------------------------------------------------------

main_ls_raw = raw_perf.loc[
    raw_perf["Portfolio"] == "Q5-Q1",
    ["Method", "mean_excess_ann", "volatility_ann", "sharpe_ratio"],
].copy()

main_ls_alpha = risk_adj.loc[
    (risk_adj["Portfolio"] == "Q5-Q1") & (risk_adj["FactorModel"] == "FF5_MOM"),
    ["Method", "alpha", "t_stat", "p_value"],
].copy()

main_ls_results = (
    main_ls_raw.merge(main_ls_alpha, on="Method", how="inner", validate="1:1")
    .sort_values("Method")
    .reset_index(drop=True)
)

main_ls_results = main_ls_results[
    ["Method", "mean_excess_ann", "volatility_ann", "sharpe_ratio", "alpha", "t_stat", "p_value"]
]


# -----------------------------------------------------------------------------
# 2) Alpha-difference table relative to baseline
# -----------------------------------------------------------------------------

alpha_difference_results = alpha_diff.copy()

if "Comparison" in alpha_difference_results.columns:
    alpha_difference_results["Comparison"] = alpha_difference_results["Comparison"].replace(
        {
            "Method2_PostMean vs Method1_Raw": "Method2_PostMean minus Method1_Raw",
            "Method3_ProbQ5 vs Method1_Raw": "Method3_ProbQ5 minus Method1_Raw",
        }
    )

wanted_comparisons = {
    "Method2_PostMean minus Method1_Raw",
    "Method3_ProbQ5 minus Method1_Raw",
    "Method2_PostMean vs Method1_Raw",
    "Method3_ProbQ5 vs Method1_Raw",
}
alpha_difference_results = alpha_difference_results[
    alpha_difference_results["Comparison"].isin(wanted_comparisons)
].copy()

alpha_difference_results = alpha_difference_results[
    ["Comparison", "alpha_difference", "t_stat", "p_value"]
].sort_values("Comparison").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3) Quintile return pattern table
# -----------------------------------------------------------------------------

quintile_pattern_results = raw_perf.loc[
    raw_perf["Portfolio"].isin(["Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"]),
    ["Method", "Portfolio", "mean_excess_ann"],
].copy()

quintile_pattern_results = (
    quintile_pattern_results
    .pivot(index="Method", columns="Portfolio", values="mean_excess_ann")
    .reset_index()
)

for col in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"]:
    if col not in quintile_pattern_results.columns:
        quintile_pattern_results[col] = pd.NA

quintile_pattern_results = quintile_pattern_results[
    ["Method", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"]
].sort_values("Method").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 4) Cumulative long-short return plot input
# -----------------------------------------------------------------------------

cumulative_ls_returns = pd.DataFrame(index=monthly_returns.index).sort_index()

for method in ["Method1_Raw", "Method2_PostMean", "Method3_ProbQ5"]:
    sub = monthly_returns[method].copy()
    ls_col = find_ls_col(sub)

    if ls_col is not None:
        ls_series = sub[ls_col].copy()
    else:
        ls_series = sub["Q5"] - sub["Q1"]

    cumulative_ls_returns[method] = (1.0 + ls_series.fillna(0.0)).cumprod() - 1.0

cumulative_ls_returns = cumulative_ls_returns.reset_index().rename(columns={"index": "Date"})

fig, ax = plt.subplots(figsize=(10, 6))
for method in ["Method1_Raw", "Method2_PostMean", "Method3_ProbQ5"]:
    ax.plot(pd.to_datetime(cumulative_ls_returns["Date"]), cumulative_ls_returns[method], label=method)
ax.set_title("Cumulative long-short returns (Q5-Q1)")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative return")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cumulative_ls_returns.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# -----------------------------------------------------------------------------
# 5) Save final outputs
# -----------------------------------------------------------------------------

main_ls_results.to_csv(OUTPUT_DIR / "main_ls_results.csv", index=False)
alpha_difference_results.to_csv(OUTPUT_DIR / "alpha_difference_results.csv", index=False)
quintile_pattern_results.to_csv(OUTPUT_DIR / "quintile_pattern_results.csv", index=False)
cumulative_ls_returns.to_csv(OUTPUT_DIR / "cumulative_ls_returns.csv", index=False)

print("\nSaved outputs:")
print(f"  {OUTPUT_DIR / 'main_ls_results.csv'}")
print(f"  {OUTPUT_DIR / 'alpha_difference_results.csv'}")
print(f"  {OUTPUT_DIR / 'quintile_pattern_results.csv'}")
print(f"  {OUTPUT_DIR / 'cumulative_ls_returns.csv'}")
print(f"  {OUTPUT_DIR / 'cumulative_ls_returns.png'}")

print("\nmain_ls_results")
print(main_ls_results)

print("\nalpha_difference_results")
print(alpha_difference_results)

print("\nquintile_pattern_results")
print(quintile_pattern_results)

print("\ncumulative_ls_returns.head()")
print(cumulative_ls_returns.head())

# -----------------------------------------------------------------------------
# 6) Save one bundle for Overleaf visualization
# -----------------------------------------------------------------------------

bundle = {
    "run_dir": str(RUN_DIR),
    "source_files": {
        "raw_performance_csv": str(RAW_PERF_CSV),
        "risk_adjusted_csv": str(RISK_ADJ_CSV),
        "alpha_differences_csv": str(ALPHA_DIFF_CSV),
        "monthly_portfolio_returns_csv": str(MONTHLY_RETURNS_CSV),
    },
    "main_ls_results": main_ls_results.to_dict(orient="records"),
    "alpha_difference_results": alpha_difference_results.to_dict(orient="records"),
    "quintile_pattern_results": quintile_pattern_results.to_dict(orient="records"),
    "cumulative_ls_returns": cumulative_ls_returns.to_dict(orient="records"),
}

bundle_path = OUTPUT_DIR / "main_results_bundle.json"
with open(bundle_path, "w", encoding="utf-8") as f:
    json.dump(bundle, f, indent=2, ensure_ascii=False, default=str)

print(f"  {bundle_path}")