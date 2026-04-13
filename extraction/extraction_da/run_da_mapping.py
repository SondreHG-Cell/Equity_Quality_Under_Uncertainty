import sys
from pathlib import Path

from openai import RateLimitError, APIStatusError

from extraction.extraction_da.llm_row_mapper_da import llm_map_da_rows, save_mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import MAPPINGS_DIR

PROF_MAP_DIR = MAPPINGS_DIR / "prof_mappings"
DA_MAP_DIR   = MAPPINGS_DIR / "da_mappings"

VALID_EXCHANGES = ["isx", "cox", "stx", "obx", "hex"]

# Choose which exchanges to run
EXCHANGES_TO_RUN = ["obx"]
# Example:
# EXCHANGES_TO_RUN = ["isx"]
# EXCHANGES_TO_RUN = ["isx", "stx", "obx"]


def run_exchange(exchange: str) -> None:
    if exchange not in VALID_EXCHANGES:
        raise ValueError(f"Invalid exchange: {exchange}. Must be one of {VALID_EXCHANGES}")

    input_dir = DATA_DIR / f"{exchange}_xlsx"
    original_mappings_dir = MAPPINGS_BASE_DIR / f"mappings_{exchange}"
    da_mappings_dir = DA_MAPPINGS_BASE_DIR / f"da_mappings_{exchange}"

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input folder for {exchange}: {input_dir}")

    if not original_mappings_dir.exists():
        raise FileNotFoundError(f"Missing original mappings folder for {exchange}: {original_mappings_dir}")

    xlsx_files = sorted(input_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found for {exchange} in: {input_dir}")

    da_mappings_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Exchange: {exchange} ===")
    print(f"Found {len(xlsx_files)} files in {input_dir}")

    n_skip_existing = 0
    n_skip_missing_mapping = 0
    n_ok = 0
    n_error = 0

    for f in xlsx_files:
        firm_id = f.stem
        original_mapping_path = original_mappings_dir / f"{firm_id}.json"
        out_path = da_mappings_dir / f"{firm_id}.json"

        if out_path.exists():
            print(f"[skip] {exchange} | {firm_id} (already mapped)")
            n_skip_existing += 1
            continue

        if not original_mapping_path.exists():
            print(f"[skip] {exchange} | {firm_id} (missing original mapping: {original_mapping_path.name})")
            n_skip_missing_mapping += 1
            continue

        print(f"[map]  {exchange} | {firm_id}")
        try:
            mapping = llm_map_da_rows(
                xlsx_path=f,
                original_mapping_path=original_mapping_path,
                model="gpt-5.4",
                reasoning_effort="low",
            )
            save_mapping(mapping, out_path)
            n_ok += 1

        except RateLimitError as e:
            print(f"[stop] {exchange} | Rate limit / quota problem: {e}")
            print("Fix billing/quota and rerun; already-mapped firms will be skipped.")
            raise

        except APIStatusError as e:
            print(f"[stop] {exchange} | OpenAI API error: {e}")
            raise

        except Exception as e:
            print(f"[error] {exchange} | {firm_id}: {type(e).__name__}: {e}")
            n_error += 1
            continue

    print(
        f"[done] {exchange} | "
        f"ok={n_ok}, skip_existing={n_skip_existing}, "
        f"skip_missing_mapping={n_skip_missing_mapping}, errors={n_error}"
    )


def main():
    invalid = [ex for ex in EXCHANGES_TO_RUN if ex not in VALID_EXCHANGES]
    if invalid:
        raise ValueError(f"Invalid exchanges in EXCHANGES_TO_RUN: {invalid}. Must be from {VALID_EXCHANGES}")

    print(f"Running exchanges: {EXCHANGES_TO_RUN}")

    for exchange in EXCHANGES_TO_RUN:
        try:
            run_exchange(exchange)
        except RateLimitError:
            return
        except APIStatusError:
            return

    print("\nAll selected exchanges done.")


if __name__ == "__main__":
    main()