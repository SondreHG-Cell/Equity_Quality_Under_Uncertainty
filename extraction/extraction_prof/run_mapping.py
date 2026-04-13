import sys
from pathlib import Path

from openai import RateLimitError, APIStatusError

from extraction.extraction_prof.llm_row_mapper import llm_map_prof_rows, save_mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import RAW_DIR, MAPPINGS_DIR as _MAPPINGS_DIR

INPUT_DIR    = RAW_DIR / "obx_xlsx"
MAPPINGS_DIR = _MAPPINGS_DIR / "prof_mappings" / "mappings_obx"


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
            mapping = llm_map_prof_rows(f, model="gpt-5.4", reasoning_effort="low")
            save_mapping(mapping, out_path)

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