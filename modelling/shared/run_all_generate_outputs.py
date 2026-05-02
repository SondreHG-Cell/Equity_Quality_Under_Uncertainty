from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


@dataclass(frozen=True)
class GenerateTask:
    name: str
    script: str
    args: tuple[str, ...] = ()
    optional: bool = False
    supports_nw_lags: bool = True


STANDARD_TASKS = [
    GenerateTask(
        name="standard_ucits_5_10_40",
        script="generate_capped_weight_risk_adjusted_table_data.py",
    ),
]


MAIN_THESIS_TASKS = [
    *STANDARD_TASKS,
    GenerateTask(
        name="q4_q2_ucits_5_10_40",
        script="generate_risk_adjusted_table_data_q4_q2.py",
    ),
    GenerateTask(
        name="size_split_ucits_5_10_40",
        script="generate_size_split_risk_adjusted_table_data.py",
    ),
    GenerateTask(
        name="sector_neutral_ucits_5_10_40",
        script="generate_sector_neutral_risk_adjusted_table_data.py",
    ),
    GenerateTask(
        name="exchange_split_ucits_5_10_40",
        script="generate_exchange_split_risk_adjusted_table_data.py",
    ),
    GenerateTask(
        name="exchange_neutral_ucits_5_10_40",
        script="generate_exchange_neutral_risk_adjusted_table_data.py",
    ),
    GenerateTask(
        name="exchange_neutral_by_exchange_ucits_5_10_40",
        script="generate_exchange_neutral_by_exchange_risk_adjusted_table_data.py",
    ),
    GenerateTask(
        name="exchange_quantile_distribution_all_exchanges",
        script="generate_exchange_quantile_distribution_diagrams.py",
        supports_nw_lags=False,
    ),
    GenerateTask(
        name="exchange_quantile_distribution_excluding_iceland",
        script="generate_exchange_quantile_distribution_diagrams.py",
        args=("--exclude-iceland",),
        supports_nw_lags=False,
    ),
]


OPTIONAL_TASKS = [
    GenerateTask(
        name="original_value_weighted",
        script="generate_risk_adjusted_table_data.py",
        optional=True,
    ),
    GenerateTask(
        name="equal_weighted",
        script="generate_equal_weighted_risk_adjusted_table_data.py",
        optional=True,
    ),
]


PROFILE_TASKS = {
    "ols-standard": STANDARD_TASKS,
    "main": MAIN_THESIS_TASKS,
    "all": [*MAIN_THESIS_TASKS, *OPTIONAL_TASKS],
}


def parse_csv_arg(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thesis generate_*.py scripts for a completed results run. "
            "Use --profile ols-standard for OLS robustness tables only."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Results run directory, for example results/current_res or results/OLS_res.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_TASKS),
        default="main",
        help=(
            "Task profile to run. 'main' runs the thesis robustness set, "
            "'ols-standard' runs only full-sample 5/10/40, and 'all' also "
            "runs original value-weighted and equal-weighted outputs."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated task names to run from the selected profile.",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma-separated task names to skip.",
    )
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=12,
        help="Newey-West/HAC lags passed to scripts that support --nw-lags.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later tasks if one task fails.",
    )
    return parser.parse_args()


def build_tasks(profile: str, only: set[str], skip: set[str]) -> list[GenerateTask]:
    tasks = PROFILE_TASKS[profile]
    known_names = {task.name for task in tasks}

    unknown_only = sorted(only - known_names)
    if unknown_only:
        raise ValueError(
            f"Unknown --only task(s) for profile '{profile}': {unknown_only}\n"
            f"Available tasks: {sorted(known_names)}"
        )

    selected = []
    for task in tasks:
        if only and task.name not in only:
            continue
        if task.name in skip:
            continue
        selected.append(task)

    if not selected:
        raise ValueError("No tasks selected.")

    return selected


def command_for_task(task: GenerateTask, args: argparse.Namespace) -> list[str]:
    script_path = SCRIPT_DIR / task.script
    if not script_path.exists():
        raise FileNotFoundError(f"Generator script does not exist: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if args.run_dir is not None:
        cmd.extend(["--run-dir", str(args.run_dir)])
    if task.supports_nw_lags:
        cmd.extend(["--nw-lags", str(args.nw_lags)])
    cmd.extend(task.args)
    return cmd


def run_task(task: GenerateTask, cmd: list[str], dry_run: bool, env: dict[str, str]) -> int:
    printable = " ".join(cmd)
    print(f"\n=== {task.name} ===")
    print(printable)

    if dry_run:
        return 0

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return int(result.returncode)


def main() -> None:
    args = parse_args()
    only = parse_csv_arg(args.only)
    skip = parse_csv_arg(args.skip)
    tasks = build_tasks(args.profile, only=only, skip=skip)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    print("Selected generate tasks:")
    for task in tasks:
        suffix = " (optional)" if task.optional else ""
        print(f"  - {task.name}{suffix}")

    failures: list[tuple[str, int]] = []
    for task in tasks:
        cmd = command_for_task(task, args)
        returncode = run_task(task=task, cmd=cmd, dry_run=args.dry_run, env=env)
        if returncode != 0:
            failures.append((task.name, returncode))
            if not args.continue_on_error:
                break

    print("\nGenerate run summary")
    if args.dry_run:
        print(f"  Dry run only. Commands printed: {len(tasks)}")
        return

    completed = len(tasks) - len(failures)
    print(f"  Completed successfully: {completed}")
    print(f"  Failed: {len(failures)}")
    for name, returncode in failures:
        print(f"    {name}: exit code {returncode}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
