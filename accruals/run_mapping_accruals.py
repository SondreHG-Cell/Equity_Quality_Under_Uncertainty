import os
from pathlib import Path
from openai import APIStatusError, RateLimitError

from llm_row_mapper_accruals import llm_map_accruals_rows, save_mapping_accruals


# -----------------------------
# Config
# -----------s------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "data" / "cox_xlsx"
MAPPINGS_DIR = BASE_DIR / "acc_mappings" / "acc_mappings_cox"


# -----------------------------
# Batch runner
# -----------------------------
def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input folder: {INPUT_DIR}")

    xlsx_files = sorted(INPUT_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_DIR}")

    print(f"Found {len(xlsx_files)} files in {INPUT_DIR}")

    for f in xlsx_files:
        firm_id = f.stem
        out_path = MAPPINGS_DIR / f"{firm_id}.json"

        if out_path.exists():
            print(f"[skip] {firm_id} (already mapped)")
            continue

        print(f"[map]  {firm_id}")
        try:
            mapping = llm_map_accruals_rows(f, model="gpt-5.4", reasoning_effort="low")
            save_mapping_accruals(mapping, out_path)

        except RateLimitError as e:
            # includes insufficient_quota / 429
            print(f"[stop] Rate limit / quota problem: {e}")
            print("Fix billing/quota and rerun; already-mapped firms will be skipped.")
            return

        except APIStatusError as e:
            print(f"[stop] OpenAI API error: {e}")
            return

        except Exception as e:
            print(f"[error] {firm_id}: {type(e).__name__}: {e}")
            # continue to next firm to avoid losing the whole run
            continue

    print("Done.")


if __name__ == "__main__":
    main()
