from __future__ import annotations

import argparse
import json
from pathlib import Path


OLD_PROBABILISTIC = "Method3_ProbabilisticQuality"
OLD_CONSERVATIVE = "Method4_ConservativeQuality"
NEW_CONSERVATIVE = "Method3_ConservativeQuality"
NEW_PROBABILISTIC = "Method4_ProbabilisticQuality"
TEMP_PROBABILISTIC = "__METHOD_LABEL_MIGRATION_PROBABILISTIC__"

TEXT_EXTENSIONS = {".csv", ".json", ".txt", ".md"}
SKIP_FILENAMES = {"method_label_migration_report.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a result folder from the old Method3/Method4 numbering "
            "to Method3=Conservative Quality and Method4=Probabilistic Quality."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Result run directory to migrate, for example results/current_res.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would change without writing them.",
    )
    return parser.parse_args()


def migrate_text(text: str) -> tuple[str, dict[str, int]]:
    counts = {
        "probabilistic_old_occurrences": text.count(OLD_PROBABILISTIC),
        "conservative_old_occurrences": text.count(OLD_CONSERVATIVE),
    }

    migrated = (
        text.replace(OLD_PROBABILISTIC, TEMP_PROBABILISTIC)
        .replace(OLD_CONSERVATIVE, NEW_CONSERVATIVE)
        .replace(TEMP_PROBABILISTIC, NEW_PROBABILISTIC)
    )
    return migrated, counts


def candidate_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in TEXT_EXTENSIONS
        and path.name not in SKIP_FILENAMES
    ]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    changed_files = []
    total_counts = {
        "probabilistic_old_occurrences": 0,
        "conservative_old_occurrences": 0,
    }

    for path in candidate_files(run_dir):
        text = path.read_text(encoding="utf-8")
        migrated, counts = migrate_text(text)
        if migrated == text:
            continue

        for key, value in counts.items():
            total_counts[key] += value

        changed_files.append(
            {
                "path": str(path),
                "probabilistic_old_occurrences": counts["probabilistic_old_occurrences"],
                "conservative_old_occurrences": counts["conservative_old_occurrences"],
            }
        )

        if not args.dry_run:
            path.write_text(migrated, encoding="utf-8")

    report = {
        "run_dir": str(run_dir),
        "dry_run": bool(args.dry_run),
        "new_numbering": {
            "method3": NEW_CONSERVATIVE,
            "method4": NEW_PROBABILISTIC,
        },
        "changed_file_count": len(changed_files),
        "total_replacements": total_counts,
        "changed_files": changed_files,
    }

    report_path = run_dir / "method_label_migration_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
