from pathlib import Path

from openai import RateLimitError, APIStatusError

from da_row_mapper import llm_map_da_rows, save_mapping

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "data" / "isx_xlsx"
ORIGINAL_MAPPINGS_DIR = BASE_DIR / "mappings" / "mappings_isx"
DA_MAPPINGS_DIR = BASE_DIR / "da_mappings" / "da_mappings_isx"


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input folder: {INPUT_DIR}")

    if not ORIGINAL_MAPPINGS_DIR.exists():
        raise FileNotFoundError(f"Missing original mappings folder: {ORIGINAL_MAPPINGS_DIR}")

    xlsx_files = sorted(INPUT_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_DIR}")

    DA_MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(xlsx_files)} files in {INPUT_DIR}")

    for f in xlsx_files:
        firm_id = f.stem
        original_mapping_path = ORIGINAL_MAPPINGS_DIR / f"{firm_id}.json"
        out_path = DA_MAPPINGS_DIR / f"{firm_id}.json"

        if out_path.exists():
            print(f"[skip] {firm_id} (already mapped)")
            continue

        if not original_mapping_path.exists():
            print(f"[skip] {firm_id} (missing original mapping: {original_mapping_path.name})")
            continue

        print(f"[map]  {firm_id}")
        try:
            mapping = llm_map_da_rows(
                xlsx_path=f,
                original_mapping_path=original_mapping_path,
                model="gpt-5.4",
                reasoning_effort="low",
            )
            save_mapping(mapping, out_path)

        except RateLimitError as e:
            print(f"[stop] Rate limit / quota problem: {e}")
            print("Fix billing/quota and rerun; already-mapped firms will be skipped.")
            return

        except APIStatusError as e:
            print(f"[stop] OpenAI API error: {e}")
            return

        except Exception as e:
            print(f"[error] {firm_id}: {type(e).__name__}: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()