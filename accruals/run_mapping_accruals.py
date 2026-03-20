import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, RateLimitError

from llm_row_mapper_accruals import llm_map_accruals_rows, save_mapping_accruals


# -----------------------------
# Config
# -----------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "data" / "stx_xlsx"
MAPPINGS_DIR = BASE_DIR / "mappings" / "mappings_stx"
TEST_FILE_LIMIT = int(os.getenv("TEST_FILE_LIMIT", "0"))


# -----------------------------
# Batch runner
# -----------------------------
def main() -> None:
    """
    Run accruals row mapping for every workbook in the input folder.
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input folder: {INPUT_DIR}")

    xlsx_files = sorted(INPUT_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_DIR}")

    if TEST_FILE_LIMIT > 0:
        xlsx_files = xlsx_files[:TEST_FILE_LIMIT]

    print(f"Found {len(xlsx_files)} files in {INPUT_DIR}")
    if TEST_FILE_LIMIT > 0:
        print(f"TEST_FILE_LIMIT is active: processing at most {TEST_FILE_LIMIT} files")

    for file_path in xlsx_files:
        firm_id = file_path.stem
        out_path = MAPPINGS_DIR / f"{firm_id}.json"

        if out_path.exists():
            print(f"[skip] {firm_id} (already mapped)")
            continue

        print(f"[map]  {firm_id}")
        try:
            mapping = llm_map_accruals_rows(file_path, model="gpt-5.4", reasoning_effort="low")
            save_mapping_accruals(mapping, out_path)

        except RateLimitError as exc:
            print(f"[stop] Rate limit / quota problem: {exc}")
            print("Fix billing/quota and rerun; already-mapped firms will be skipped.")
            return

        except APIStatusError as exc:
            print(f"[stop] OpenAI API error: {exc}")
            return

        except Exception as exc:
            print(f"[error] {firm_id}: {type(exc).__name__}: {exc}")
            # Continue so that one bad file does not stop the whole batch.
            continue

    print("Done.")


if __name__ == "__main__":
    main()
