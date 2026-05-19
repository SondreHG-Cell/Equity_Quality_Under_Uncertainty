from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

OBSERVED_METHOD = "Method1_ObservedQuality"
METHOD_LABELS = {
    "Method1_ObservedQuality": "Observed Quality",
    "Method2_LatentQuality": "Latent Quality",
    "Method3_ConservativeQuality": "Conservative Quality",
    "Method4_ProbabilisticQuality": "Probabilistic Quality",
}
COMPARISON_METHODS = [
    "Method2_LatentQuality",
    "Method3_ConservativeQuality",
    "Method4_ProbabilisticQuality",
]
PORTFOLIOS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
FACTOR_ORDER = ["CAPM", "FF3", "FF3+MOM", "FF5", "FF5+MOM"]
STRATEGY_ORDER = ["LongShort", "Q5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate reassignment outputs and gamma/kappa sensitivity tables."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "current_res",
    )
    parser.add_argument(
        "--manual-figure-dir",
        type=Path,
        default=PROJECT_ROOT / "manual_review" / "Figures" / "result_plots",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def portfolio_col(method: str) -> str:
    return f"{method}_Portfolio"


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}\\%"


def format_num(value: float) -> str:
    return f"{value:.2f}"


def format_p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_alpha(value: float, p_value: float | None = None) -> str:
    text = format_pct(value)
    if p_value is None:
        return text
    marker = stars(float(p_value))
    if not marker:
        return text
    return f"\\dmstar{{{text}}}{{{marker}}}"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def compute_reassignment(assignments: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    by_year_rows: list[dict[str, object]] = []
    observed_col = portfolio_col(OBSERVED_METHOD)

    for method in methods:
        comp_col = portfolio_col(method)
        label = METHOD_LABELS[method]
        for observed_tail in ["Q1", "Q5"]:
            observed_mask = assignments[observed_col].eq(observed_tail)
            comparison_tail_mask = assignments[comp_col].eq(observed_tail)
            stayed_mask = observed_mask & comparison_tail_mask
            left_mask = observed_mask & ~comparison_tail_mask
            entered_mask = ~observed_mask & comparison_tail_mask
            changed_mask = left_mask | entered_mask
            opposite_tail = "Q5" if observed_tail == "Q1" else "Q1"
            direct_opposite_mask = observed_mask & assignments[comp_col].eq(opposite_tail)

            observed_n = int(observed_mask.sum())
            comparison_n = int(comparison_tail_mask.sum())
            stayed_n = int(stayed_mask.sum())
            left_n = int(left_mask.sum())
            entered_n = int(entered_mask.sum())
            changed_n = int(changed_mask.sum())
            direct_opposite_n = int(direct_opposite_mask.sum())
            rows.append(
                {
                    "ComparisonMethod": method,
                    "ComparisonMethodLabel": label,
                    "ObservedPortfolio": observed_tail,
                    "ObservedTailFirmYears": observed_n,
                    "ComparisonTailFirmYears": comparison_n,
                    "StayedInSameTailFirmYears": stayed_n,
                    "LeftObservedTailFirmYears": left_n,
                    "EnteredComparisonTailFirmYears": entered_n,
                    "TailMembershipChangedFirmYears": changed_n,
                    "DirectOppositeTailFirmYears": direct_opposite_n,
                    "ShareObservedTailStayed": stayed_n / observed_n if observed_n else np.nan,
                    "ShareObservedTailLeft": left_n / observed_n if observed_n else np.nan,
                    "ShareComparisonTailEnteredFromNonObservedTail": entered_n / comparison_n if comparison_n else np.nan,
                    "UniqueFirmsLeftObservedTail": int(assignments.loc[left_mask, "Ticker"].nunique()),
                    "UniqueFirmsEnteredComparisonTail": int(assignments.loc[entered_mask, "Ticker"].nunique()),
                }
            )

            sub = assignments.loc[observed_mask, [comp_col]].copy()
            counts = sub[comp_col].value_counts(dropna=False)
            for portfolio in PORTFOLIOS:
                count = int(counts.get(portfolio, 0))
                transition_rows.append(
                    {
                        "ComparisonMethod": method,
                        "ComparisonMethodLabel": label,
                        "ObservedPortfolio": observed_tail,
                        "ComparisonPortfolio": portfolio,
                        "FirmYears": count,
                        "ShareObservedTail": count / observed_n if observed_n else np.nan,
                    }
                )

            for year, year_df in assignments.groupby("FormationYear", sort=True):
                year_observed = year_df[observed_col].eq(observed_tail)
                year_comparison = year_df[comp_col].eq(observed_tail)
                year_observed_n = int(year_observed.sum())
                year_left_n = int((year_observed & ~year_comparison).sum())
                by_year_rows.append(
                    {
                        "FormationYear": year,
                        "ComparisonMethod": method,
                        "ComparisonMethodLabel": label,
                        "ObservedPortfolio": observed_tail,
                        "ObservedTailFirmYears": year_observed_n,
                        "LeftObservedTailFirmYears": year_left_n,
                        "ShareObservedTailLeft": year_left_n / year_observed_n if year_observed_n else np.nan,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(transition_rows), pd.DataFrame(by_year_rows)


def compute_all_portfolio_reassignment(assignments: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    observed_col = portfolio_col(OBSERVED_METHOD)
    rows: list[dict[str, object]] = []
    for method in methods:
        comp_col = portfolio_col(method)
        for portfolio in PORTFOLIOS:
            observed_mask = assignments[observed_col].eq(portfolio)
            comp_mask = assignments[comp_col].eq(portfolio)
            left_mask = observed_mask & ~comp_mask
            entered_mask = ~observed_mask & comp_mask
            observed_n = int(observed_mask.sum())
            comparison_n = int(comp_mask.sum())
            left_n = int(left_mask.sum())
            entered_n = int(entered_mask.sum())
            rows.append(
                {
                    "ComparisonMethod": method,
                    "ComparisonMethodLabel": METHOD_LABELS[method],
                    "ObservedPortfolio": portfolio,
                    "ObservedPortfolioFirmYears": observed_n,
                    "ComparisonPortfolioFirmYears": comparison_n,
                    "LeftObservedPortfolioFirmYears": left_n,
                    "EnteredComparisonPortfolioFirmYears": entered_n,
                    "ShareObservedPortfolioLeft": left_n / observed_n if observed_n else np.nan,
                    "ShareComparisonPortfolioEntered": entered_n / comparison_n if comparison_n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_main_reassignment(summary: pd.DataFrame, transition: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = COMPARISON_METHODS
    method_labels = [METHOD_LABELS[m].replace(" Quality", "") for m in methods]
    colors = {
        "Latent": "#2c73ad",
        "Conservative": "#58a7df",
        "Probabilistic": "#8ec5e8",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#f6f9fc")
        ax.grid(axis="y", color="#9ebfda", linewidth=0.6, linestyle="--", alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#a7bfd5")
        ax.spines["bottom"].set_color("#a7bfd5")
        ax.tick_params(colors="#20384f")

    x = np.arange(len(methods))
    width = 0.34
    for offset, portfolio in [(-width / 2, "Q1"), (width / 2, "Q5")]:
        vals = []
        for method in methods:
            row = summary.loc[
                summary["ComparisonMethod"].eq(method) & summary["ObservedPortfolio"].eq(portfolio)
            ].iloc[0]
            vals.append(100.0 * float(row["ShareObservedTailLeft"]))
        bars = axes[0].bar(x + offset, vals, width=width, label=portfolio, color="#2c73ad" if portfolio == "Q1" else "#8ec5e8")
        axes[0].bar_label(bars, labels=[f"{v:.1f}%" for v in vals], fontsize=8, padding=2, color="#173a5e")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels)
    axes[0].set_ylabel("Share reassigned (%)", color="#173a5e")
    axes[0].set_title("Panel A: Tail reassignment rates", color="#173a5e", fontweight="bold", fontsize=10)
    axes[0].legend(frameon=False)

    labels = []
    bases = np.zeros(len(methods) * 2)
    positions = np.arange(len(methods) * 2)
    for method in methods:
        labels.extend([f"{METHOD_LABELS[method].replace(' Quality', '')}\nQ1", f"{METHOD_LABELS[method].replace(' Quality', '')}\nQ5"])

    destination_colors = {
        "Q1": "#173a5e",
        "Q2": "#2c73ad",
        "Q3": "#58a7df",
        "Q4": "#8ec5e8",
        "Q5": "#c3ddf0",
    }
    for dest in PORTFOLIOS:
        vals = []
        for method in methods:
            for observed_tail in ["Q1", "Q5"]:
                if dest == observed_tail:
                    vals.append(0.0)
                    continue
                row = transition.loc[
                    transition["ComparisonMethod"].eq(method)
                    & transition["ObservedPortfolio"].eq(observed_tail)
                    & transition["ComparisonPortfolio"].eq(dest)
                ].iloc[0]
                vals.append(100.0 * float(row["ShareObservedTail"]))
        axes[1].bar(positions, vals, bottom=bases, color=destination_colors[dest], label=dest)
        bases += np.array(vals)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Share of observed tail (%)", color="#173a5e")
    axes[1].set_title("Panel B: Destination portfolios", color="#173a5e", fontweight="bold", fontsize=10)
    axes[1].legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.suptitle("Q1 and Q5 reassignment relative to Observed Quality", color="#173a5e", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])

    for name in ["q1_q5_reassignment_vs_observed_compact", "q1_q5_reassignment_vs_observed_figure"]:
        fig.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_main_reassignment(run_dir: Path, manual_figure_dir: Path) -> None:
    assignment_path = run_dir / "portfolio_formation" / "portfolio_assignments_wide.csv"
    assignments = pd.read_csv(assignment_path)
    output_dir = run_dir / "portfolio_formation" / "quantile_reassignment_vs_observed"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, transition, by_year = compute_reassignment(assignments, COMPARISON_METHODS)
    all_summary = compute_all_portfolio_reassignment(assignments, COMPARISON_METHODS)
    summary.to_csv(output_dir / "q1_q5_reassignment_summary_vs_observed.csv", index=False)
    transition.to_csv(output_dir / "q1_q5_transition_matrix_vs_observed.csv", index=False)
    by_year.to_csv(output_dir / "q1_q5_reassignment_summary_vs_observed_by_year.csv", index=False)
    all_summary.to_csv(output_dir / "all_portfolio_reassignment_summary_vs_observed.csv", index=False)

    plot_main_reassignment(summary, transition, output_dir)
    manual_figure_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "q1_q5_reassignment_vs_observed_compact.png",
        "q1_q5_reassignment_vs_observed_compact.pdf",
        "q1_q5_reassignment_vs_observed_figure.png",
        "q1_q5_reassignment_vs_observed_figure.pdf",
    ]:
        shutil.copy2(output_dir / name, manual_figure_dir / name)


def variant_dir(portfolio_eval_dir: Path, kind: str, value: str) -> Path:
    label = f"{float(value):.2f}".replace(".", "_")
    if kind == "kappa":
        return portfolio_eval_dir / f"thesis_risk_adjusted_tables_latent_kappa_{label}_ucits_5_10_40"
    if kind == "gamma":
        return portfolio_eval_dir / f"thesis_risk_adjusted_tables_conservative_gamma_{label}_ucits_5_10_40"
    if kind == "prob_kappa":
        return portfolio_eval_dir / f"thesis_risk_adjusted_tables_probabilistic_kappa_{label}_ucits_5_10_40"
    raise ValueError(kind)


def param_col_for_kind(kind: str) -> str:
    if kind == "kappa":
        return "Kappa"
    if kind == "gamma":
        return "Gamma"
    if kind == "prob_kappa":
        return "KappaP"
    raise ValueError(kind)


def param_macro_for_kind(kind: str) -> str:
    if kind == "kappa":
        return "\\kappa"
    if kind == "gamma":
        return "\\gamma"
    if kind == "prob_kappa":
        return "\\kappa_P"
    raise ValueError(kind)


def collect_variant_outputs(
    portfolio_eval_dir: Path,
    kind: str,
    values: list[str],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    param_col = param_col_for_kind(kind)
    raw_rows = []
    alpha_rows = []
    diff_rows = []
    reassignment_rows = []
    all_reassignment_rows = []
    transition_rows = []

    for value in values:
        table_dir = variant_dir(portfolio_eval_dir, kind, value)
        param_value = float(value)
        raw = pd.read_csv(table_dir / "table_raw_performance.csv")
        raw = raw.loc[raw["Method"].isin([OBSERVED_METHOD, method])].copy()
        raw[param_col] = param_value
        raw_rows.append(raw)

        for file_name in ["table_ls_alpha_levels.csv", "table_q5_alpha_levels.csv"]:
            alpha = pd.read_csv(table_dir / file_name)
            alpha = alpha.loc[alpha["Method"].eq(method)].copy()
            alpha[param_col] = param_value
            alpha_rows.append(alpha)

        for file_name in ["table_ls_alpha_differences.csv", "table_q5_alpha_differences.csv"]:
            diff = pd.read_csv(table_dir / file_name)
            diff = diff.loc[diff["Comparison"].str.contains(method, regex=False)].copy()
            diff[param_col] = param_value
            diff_rows.append(diff)

        assignment_path = table_dir / "inputs" / "portfolio_formation" / "portfolio_assignments_wide.csv"
        assignments = pd.read_csv(assignment_path)
        summary, transition, _ = compute_reassignment(assignments, [method])
        all_summary = compute_all_portfolio_reassignment(assignments, [method])
        summary.insert(0, param_col, param_value)
        transition.insert(0, param_col, param_value)
        all_summary.insert(0, param_col, param_value)
        reassignment_rows.append(summary)
        transition_rows.append(transition)
        all_reassignment_rows.append(all_summary)

    return (
        pd.concat(raw_rows, ignore_index=True),
        pd.concat(alpha_rows, ignore_index=True),
        pd.concat(diff_rows, ignore_index=True),
        pd.concat(reassignment_rows, ignore_index=True),
        pd.concat(transition_rows, ignore_index=True),
        pd.concat(all_reassignment_rows, ignore_index=True),
    )


def alpha_table_tex(
    alpha: pd.DataFrame,
    diffs: pd.DataFrame,
    values: list[str],
    param_macro: str,
    strategy: str,
    caption: str,
    label: str,
    main_value: str,
    method_name: str,
) -> str:
    alpha = alpha.loc[alpha["PortfolioStrategy"].eq(strategy)].copy()
    diffs = diffs.loc[diffs["PortfolioStrategy"].eq(strategy)].copy()
    value_floats = [float(v) for v in values]
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * 3}-{4 + i * 3}}}" for i in range(len(values))
    )
    colspec = "l" + "ccc" * len(values)
    header = " & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{${param_macro}={float(v):.2f}$}}" for v in values
    ) + " \\\\"
    columns = "Model" + " & $\\alpha$ & $t$-stat & $p$-value" * len(values) + " \\\\"

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{threeparttable}",
        "{\\centering\\textbf{Panel A: Annualised alphas}\\par}",
        "\\vspace{0.4em}",
        f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}",
        "\\toprule",
        header,
        cmidrules,
        columns,
        "\\midrule",
    ]
    for model in FACTOR_ORDER:
        row = [model.ljust(9)]
        for value in value_floats:
            match = alpha.loc[(alpha["FactorModel"].eq(model)) & np.isclose(alpha[param_macro_name(param_macro)], value)]
            if match.empty:
                row.extend(["", "", ""])
            else:
                rec = match.iloc[0]
                row.extend(
                    [
                        format_alpha(float(rec["alpha_annualized"]), float(rec["p_value"])),
                        format_num(float(rec["t_stat"])),
                        format_p(float(rec["p_value"])),
                    ]
                )
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\vspace{1em}",
            "{\\centering\\textbf{Panel B: Annualised alpha differences relative to Observed Quality}\\par}",
            "\\vspace{0.4em}",
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}",
            "\\toprule",
            header,
            cmidrules,
            "Model" + " & $\\Delta\\alpha$ & $t$-stat & $p$-value" * len(values) + " \\\\",
            "\\midrule",
        ]
    )
    for model in FACTOR_ORDER:
        row = [model.ljust(9)]
        for value in value_floats:
            match = diffs.loc[(diffs["FactorModel"].eq(model)) & np.isclose(diffs[param_macro_name(param_macro)], value)]
            if match.empty:
                row.extend(["", "", ""])
            else:
                rec = match.iloc[0]
                row.extend(
                    [
                        format_pct(float(rec["alpha_difference_annualized"])),
                        format_num(float(rec["t_stat"])),
                        format_p(float(rec["p_value"])),
                    ]
                )
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            (
                "\\item Notes: All alphas and alpha differences are estimated from monthly regressions and "
                "annualised by multiplying monthly estimates by 12. $t$-statistics and $p$-values are based "
                "on Newey--West standard errors with 12 lags. Portfolio weights are subject to a UCITS-inspired "
                f"5/10/40 concentration cap. The ${param_macro}={float(main_value):.2f}$ specification corresponds "
                f"to the main {method_name} specification. Significance levels are denoted by "
                "$^{*} p<0.10$, $^{**} p<0.05$, and $^{***} p<0.01$."
            ),
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def param_macro_name(param_macro: str) -> str:
    if param_macro == "\\kappa":
        return "Kappa"
    if param_macro == "\\gamma":
        return "Gamma"
    if param_macro == "\\kappa_P":
        return "KappaP"
    raise ValueError(param_macro)


def raw_performance_tex(
    raw: pd.DataFrame,
    values: list[str],
    param_macro: str,
    caption: str,
    label: str,
    method: str,
    main_value: str,
    method_name: str,
) -> str:
    param_col = param_macro_name(param_macro)
    rows_by_strategy = {}
    for strategy in STRATEGY_ORDER:
        strategy_rows = []
        for value in [float(v) for v in values]:
            match = raw.loc[
                raw["PortfolioStrategy"].eq(strategy)
                & raw["Method"].eq(method)
                & np.isclose(raw[param_col], value)
            ]
            if match.empty:
                continue
            rec = match.iloc[0]
            strategy_rows.append(
                [
                    f"{method_name} (${param_macro}={value:.2f}$)",
                    format_pct(float(rec["annualized_return"])),
                    format_pct(float(rec["volatility_ann"])),
                    format_num(float(rec["sharpe_ratio"])),
                    format_pct(float(rec["max_drawdown"])),
                ]
            )
        rows_by_strategy[strategy] = strategy_rows

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption[{caption}]{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{threeparttable}",
    ]
    for idx, (strategy, panel_title) in enumerate([("LongShort", "Panel A: Long--short"), ("Q5", "Panel B: Long")]):
        if idx:
            lines.append("\\vspace{1em}")
        lines.extend(
            [
                f"{{\\centering\\textbf{{{panel_title}}}\\par}}",
                "\\vspace{0.4em}",
                "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc}",
                "\\toprule",
                "Method & Ann. ret. & Ann. vol. & Sharpe & Max DD \\\\",
                "\\midrule",
            ]
        )
        for row in rows_by_strategy[strategy]:
            lines.append(" & ".join(row) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular*}"])

    lines.extend(
        [
            "\\begin{tablenotes}[flushleft]",
            "\\scriptsize",
            (
                "\\item Note: Returns and volatility are annualised from monthly returns. The Sharpe ratio is "
                "computed using excess returns. Max DD denotes the maximum drawdown over the sample period. "
                f"The ${param_macro}={float(main_value):.2f}$ specification corresponds to the main {method_name} specification."
            ),
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def reassignment_tex(
    summary: pd.DataFrame,
    values: list[str],
    param_macro: str,
    caption: str,
    label: str,
    main_value: str,
    method_name: str,
) -> str:
    param_col = param_macro_name(param_macro)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption[{caption}]{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{threeparttable}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}",
        "\\toprule",
        f"${param_macro}$ & Portfolio & Observed tail & Same tail & Left & Entered & Reassigned share \\\\",
        "\\midrule",
    ]
    for value in [float(v) for v in values]:
        for portfolio in ["Q1", "Q5"]:
            rec = summary.loc[
                np.isclose(summary[param_col], value) & summary["ObservedPortfolio"].eq(portfolio)
            ].iloc[0]
            lines.append(
                f"{value:.2f} & {portfolio} & {int(rec['ObservedTailFirmYears']):,} & "
                f"{int(rec['StayedInSameTailFirmYears']):,} & {int(rec['LeftObservedTailFirmYears']):,} & "
                f"{int(rec['EnteredComparisonTailFirmYears']):,} & "
                f"{100.0 * float(rec['ShareObservedTailLeft']):.2f}\\% \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\begin{tablenotes}[flushleft]",
            "\\scriptsize",
            (
                "\\item Note: The table reports reassignment rates relative to the Observed Quality portfolios. "
                "\\emph{Left} denotes observed Q1 or Q5 firm-years that are no longer assigned to the same tail "
                f"portfolio under {method_name}. \\emph{{Entered}} denotes firm-years entering the corresponding "
                f"{method_name} tail portfolio from outside the observed tail. The ${param_macro}={float(main_value):.2f}$ "
                f"specification corresponds to the main {method_name} specification."
            ),
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_sensitivity_outputs(
    portfolio_eval_dir: Path,
    kind: str,
    values: list[str],
    method: str,
    main_value: str,
) -> None:
    param_col = param_col_for_kind(kind)
    param_macro = param_macro_for_kind(kind)
    method_name = METHOD_LABELS[method]
    if kind == "kappa":
        prefix = "latent_kappa"
        out_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_latent_kappa_robustness_ucits_5_10_40"
        readable = "Latent Quality kappa robustness"
    elif kind == "gamma":
        prefix = "conservative_gamma"
        out_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_conservative_gamma_robustness_ucits_5_10_40"
        readable = "Conservative Quality gamma robustness"
    elif kind == "prob_kappa":
        prefix = "probabilistic_kappa"
        out_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_probabilistic_kappa_robustness_ucits_5_10_40"
        readable = "Probabilistic Quality $\\kappa_P$ robustness"
    else:
        raise ValueError(kind)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, alpha, diffs, reassignment, transition, all_reassignment = collect_variant_outputs(
        portfolio_eval_dir=portfolio_eval_dir,
        kind=kind,
        values=values,
        method=method,
    )
    raw.to_csv(out_dir / f"{prefix}_raw_performance.csv", index=False)
    alpha.to_csv(out_dir / f"{prefix}_alpha_levels.csv", index=False)
    diffs.to_csv(out_dir / f"{prefix}_alpha_differences.csv", index=False)
    reassignment.to_csv(out_dir / f"{prefix}_q1_q5_reassignment_summary_vs_observed.csv", index=False)
    transition.to_csv(out_dir / f"{prefix}_q1_q5_transition_matrix_vs_observed.csv", index=False)
    all_reassignment.to_csv(out_dir / f"{prefix}_all_portfolio_reassignment_summary_vs_observed.csv", index=False)

    (out_dir / f"{prefix}_raw_performance.tex").write_text(
        raw_performance_tex(
            raw=raw,
            values=values,
            param_macro=param_macro,
            caption=f"Unadjusted performance: {readable}",
            label=f"tab:{prefix}_raw_performance",
            method=method,
            main_value=main_value,
            method_name=method_name,
        )
    )
    (out_dir / f"{prefix}_ls_alpha_results.tex").write_text(
        alpha_table_tex(
            alpha=alpha,
            diffs=diffs,
            values=values,
            param_macro=param_macro,
            strategy="LongShort",
            caption=f"Risk-adjusted performance of the long--short strategy: {readable}",
            label=f"tab:{prefix}_ls_alpha_results",
            main_value=main_value,
            method_name=method_name,
        )
    )
    (out_dir / f"{prefix}_q5_alpha_results.tex").write_text(
        alpha_table_tex(
            alpha=alpha,
            diffs=diffs,
            values=values,
            param_macro=param_macro,
            strategy="Q5",
            caption=f"Risk-adjusted performance of the long strategy: {readable}",
            label=f"tab:{prefix}_q5_alpha_results",
            main_value=main_value,
            method_name=method_name,
        )
    )
    (out_dir / f"{prefix}_reassignment.tex").write_text(
        reassignment_tex(
            summary=reassignment,
            values=values,
            param_macro=param_macro,
            caption=f"Q1 and Q5 reassignment rates for {readable}",
            label=f"tab:{prefix}_reassignment",
            main_value=main_value,
            method_name=method_name,
        )
    )
    print(f"Updated {kind} sensitivity outputs in {out_dir}")


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    portfolio_eval_dir = run_dir / "portfolio_evaluation"
    write_main_reassignment(run_dir=run_dir, manual_figure_dir=resolve(args.manual_figure_dir))
    write_sensitivity_outputs(
        portfolio_eval_dir=portfolio_eval_dir,
        kind="kappa",
        values=["0.12", "0.08", "0.06", "0.04"],
        method="Method2_LatentQuality",
        main_value="0.06",
    )
    write_sensitivity_outputs(
        portfolio_eval_dir=portfolio_eval_dir,
        kind="gamma",
        values=["0.15", "0.20", "0.30", "0.40", "0.50"],
        method="Method3_ConservativeQuality",
        main_value="0.40",
    )
    write_sensitivity_outputs(
        portfolio_eval_dir=portfolio_eval_dir,
        kind="prob_kappa",
        values=["0.04", "0.06", "0.08", "0.10", "0.12"],
        method="Method4_ProbabilisticQuality",
        main_value="0.06",
    )
    print("Reassignment and sensitivity outputs regenerated.")


if __name__ == "__main__":
    main()
