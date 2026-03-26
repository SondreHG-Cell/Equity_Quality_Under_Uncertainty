from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"  # "low" is also reasonable for testing

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# -----------------------------
# D&A targets
# -----------------------------
DA_TARGETS = {
    "COGS_DA": (
        "Depreciation and/or amortisation related to COGS / cost of sales / cost of revenues. "
        "Include ONLY separate D&A rows that should be added later, relative to the selected COGS final_choice rows "
        "from the original JSON. If D&A appears already embedded in the selected COGS row(s), use final_choice=[]. "
        "If both a D&A summary row and its components exist, prefer the summary row only."
    ),
    "XSGA_DA": (
        "Depreciation and/or amortisation related to SG&A / operating overhead / admin / marketing / R&D. "
        "Include ONLY separate D&A rows that should be added later, relative to the selected XSGA_COMPONENTS "
        "final_choice rows from the original JSON. If D&A appears already embedded in the selected XSGA_COMPONENTS "
        "row(s), use final_choice=[]. If both a D&A summary row and its components exist, prefer the summary row only."
    ),
}


# -----------------------------
# Shared helpers
# -----------------------------
@dataclass
class ShortlistRule:
    include: List[str]
    exclude: List[str]


def _contains_any(s: str, needles: List[str]) -> bool:
    s2 = s.lower()
    return any(n.lower() in s2 for n in needles)


# -----------------------------
# D&A shortlist rules
# -----------------------------
DA_SHORTLIST_RULES = {
    "COGS_DA": ShortlistRule(
        include=[
            "depreciation",
            "amortization",
            "amortisation",
            "depreciation in cogs",
            "amortization in cogs",
            "amortisation in cogs",
            "depreciation in cost of sales",
            "amortization in cost of sales",
            "amortisation in cost of sales",
            "depreciation in cor/cogs",
            "amortization in cor/cogs",
            "amortisation in cor/cogs",
            "depreciation in cost of revenue",
            "amortization in cost of revenue",
            "amortisation in cost of revenue",
            "depreciation in production",
            "amortization in production",
            "amortisation in production",
            "depreciation in inventory",
            "amortization in inventory",
            "amortisation in inventory",
        ],
        exclude=[
            "impairment",
            "interest",
            "tax",
            "finance",
            "financial",
            "fair value",
            "derivative",
            "foreign exchange",
            "fx",
        ],
    ),
    "XSGA_DA": ShortlistRule(
        include=[
            "depreciation",
            "amortization",
            "amortisation",
            "depreciation in sga",
            "amortization in sga",
            "amortisation in sga",
            "depreciation in administrative",
            "amortization in administrative",
            "amortisation in administrative",
            "depreciation in admin",
            "amortization in admin",
            "amortisation in admin",
            "depreciation in marketing",
            "amortization in marketing",
            "amortisation in marketing",
            "depreciation in r&d",
            "amortization in r&d",
            "amortisation in r&d",
            "depreciation in research",
            "amortization in research",
            "amortisation in research",
            "amortization in operating expenses",
            "amortisation in operating expenses",
            "depreciation in operating expenses",
        ],
        exclude=[
            "impairment",
            "interest",
            "tax",
            "finance",
            "financial",
            "fair value",
            "derivative",
            "foreign exchange",
            "fx",
        ],
    ),
}


# -----------------------------
# Output schema
# -----------------------------
DA_MAPPING_SCHEMA = {
    "name": "da_row_mapping_v1",
    "schema": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "enum": list(DA_TARGETS.keys())},
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sheet_name": {"type": "string", "minLength": 1},
                                    "row_label": {"type": "string", "minLength": 1},
                                    "why": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["sheet_name", "row_label", "why", "confidence"],
                                "additionalProperties": False,
                            },
                        },
                        "final_choice": {
                            "type": "array",
                            "minItems": 0,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sheet_name": {"type": "string", "minLength": 1},
                                    "row_label": {"type": "string", "minLength": 1},
                                },
                                "required": ["sheet_name", "row_label"],
                                "additionalProperties": False,
                            },
                        },
                        "needs_manual_review": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": ["variable", "candidates", "final_choice", "needs_manual_review", "notes"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["variables"],
        "additionalProperties": False,
    },
}


# -----------------------------
# Read the original PROF mapping
# -----------------------------
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_existing_cogs_xsga_choices(mapping: dict) -> dict:
    out = {"variables": []}

    for var in ["COGS", "XSGA_COMPONENTS", "XRD"]:
        found = None
        for block in mapping.get("variables", []):
            if block.get("variable") == var:
                found = block
                break

        if found is None:
            out["variables"].append(
                {
                    "variable": var,
                    "final_choice": [],
                    "notes": f"{var} not found in original mapping.",
                }
            )
        else:
            out["variables"].append(
                {
                    "variable": var,
                    "final_choice": found.get("final_choice", []),
                    "notes": found.get("notes", ""),
                }
            )

    return out


# -----------------------------
# Read income statement excerpt
# -----------------------------
def read_income_statement_excerpt_a_to_f(
    xlsx_path: Path,
    max_rows: int = 5000,
    min_preview_rows: int = 25,
) -> tuple[str, list[list[str]]]:
    """
    Reads A:F from the income statement sheet only.

    Start rule:
    - find the row where column A is exactly 'Field Name'
    - start the preview from the next row

    Cut rule:
    - find the first row whose label contains 'interest'
    - if that row appears before `min_preview_rows`, cut at `min_preview_rows`
    - otherwise cut at the first 'interest' row
    - if no 'interest' row exists, keep full preview
    """
    xl = pd.ExcelFile(xlsx_path)
    sheet_names = xl.sheet_names

    income_sheet = None
    for s in sheet_names:
        low = s.lower()
        if any(k in low for k in ["income", "result", "profit", "statement"]):
            income_sheet = s
            break

    if income_sheet is None:
        income_sheet = sheet_names[0]

    # Read raw with no fixed skiprows
    df = pd.read_excel(
        xlsx_path,
        sheet_name=income_sheet,
        usecols="A:F",
        header=None,
        nrows=max_rows,
    )

    if df.shape[1] == 0:
        return income_sheet, []

    # Clean everything to strings
    df = df.fillna("")
    df = df.astype(str).apply(lambda col: col.str.strip())

    # Find header row where column A == "Field Name"
    header_idx = None
    for i in range(len(df)):
        if df.iat[i, 0] == "Field Name":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"'Field Name' not found in column A for sheet {income_sheet} in {xlsx_path.name}")

    # Data starts after the Field Name row
    data_df = df.iloc[header_idx + 1 :].copy()

    preview_rows: list[list[str]] = []
    for _, row in data_df.iterrows():
        row_vals = [str(row.iloc[j]).strip() if j < len(row) else "" for j in range(min(6, len(row)))]
        while len(row_vals) < 6:
            row_vals.append("")

        if row_vals[0] != "":
            preview_rows.append(row_vals)

    first_interest_idx = None
    for i, r in enumerate(preview_rows):
        label = r[0].lower()
        if "interest" in label:
            first_interest_idx = i
            break

    if first_interest_idx is not None:
        cut_idx = max(first_interest_idx, min_preview_rows)
        preview_rows = preview_rows[:cut_idx]

    return income_sheet, preview_rows


# -----------------------------
# Candidate shortlisting
# -----------------------------
def shortlist_da_candidates(
    income_sheet_name: str,
    income_preview: list[list[str]],
    variable: str,
    max_rows: int = 40,
) -> list[tuple[str, str]]:
    rule = DA_SHORTLIST_RULES[variable]
    out: list[tuple[str, str]] = []

    labels = [r[0] for r in income_preview if r and r[0].strip() != ""]

    for lab in labels:
        lo = lab.lower()
        if _contains_any(lo, rule.include) and not _contains_any(lo, rule.exclude):
            out.append((income_sheet_name, lab))

    seen = set()
    deduped = []
    for sheet, lab in out:
        key = (sheet, lab)
        if key not in seen:
            seen.add(key)
            deduped.append((sheet, lab))

    return deduped[:max_rows]


# -----------------------------
# Prompt
# -----------------------------
def build_da_prompt(
    income_sheet_name: str,
    income_preview: list[list[str]],
    existing_choices: dict,
    shortlists: dict[str, list[tuple[str, str]]],
) -> str:
    lines: list[str] = []

    lines.append("You are auditing whether depreciation and amortisation (D&A) must be added separately to previously selected PROF rows.")
    lines.append("Return JSON ONLY that conforms to the given schema.")
    lines.append("")
    lines.append("You are given:")
    lines.append("1) the already selected final_choice rows for COGS, XSGA_COMPONENTS, and XRD from an earlier mapping step")
    lines.append("2) the income statement preview only, truncated at operating profit / EBIT / profit before financial items")
    lines.append("")
    lines.append("Goal:")
    lines.append("- For COGS_DA: choose ONLY D&A rows that should be ADDED separately later to COGS.")
    lines.append("- For XSGA_DA: choose ONLY D&A rows that should be ADDED separately later to XSGA_COMPONENTS.")
    lines.append("- If D&A is already embedded in the selected parent row(s), then final_choice must be [].")
    lines.append("")
    
    lines.append("Important scope restriction:")
    lines.append("- The preview is intended to show only the operating section before financial lines begin.")
    lines.append("- You must focus only on rows BEFORE financial lines.")
    lines.append("- If a row appears to belong to interest income, interest expense, finance costs, financial income/expenses, FX, fair value, tax, or other non-operating items, ignore it even if it appears in the preview.")
    lines.append("- For this task, only operating rows above the financial section are relevant.")
    lines.append("")

    lines.append("Critical rules:")
    lines.append("1) Choose row labels EXACTLY as written in column A.")
    lines.append("2) Do NOT invent labels.")
    lines.append("3) Use ONLY rows from the provided income statement excerpt.")
    lines.append("4) Ignore finance, tax, fair value, FX, derivative, impairment, and non-operating rows.")
    lines.append("5) Avoid double counting.")
    lines.append("6) Set needs_manual_review=true ONLY when you are genuinely uncertain.")
    lines.append("7) Do NOT set needs_manual_review=true merely because you choose a row outside the candidate shortlist.")
    lines.append("")

    lines.append("Reference-bucket rule:")
    lines.append("- The existing final_choice rows in the JSON define the reference buckets.")
    lines.append("- For COGS_DA, evaluate candidate D&A rows relative to the selected final_choice row(s) for COGS.")
    lines.append("- For XSGA_DA, evaluate candidate D&A rows relative to the selected final_choice row(s) for XSGA_COMPONENTS.")
    lines.append("- A D&A row should be added only if it appears to sit outside, alongside, or separately from those selected final_choice rows.")
    lines.append("- If the D&A row appears to be a sub-component or contained part of a selected final_choice row, do not add it separately.")
    lines.append("")

    lines.append("Definition of the selected reference buckets:")
    lines.append("- The selected COGS bucket is the sum of the rows listed in the COGS final_choice from the original JSON.")
    lines.append("- The selected XSGA_COMPONENTS bucket is the sum of the rows listed in the XSGA_COMPONENTS final_choice from the original JSON.")
    lines.append("- Therefore, when deciding whether a D&A row is already included, compare it to the combined selected bucket for that variable, not just to one selected row in isolation.")
    lines.append("- A D&A row should be added only if it appears to sit outside or separately from that combined selected bucket.")
    lines.append("")

    lines.append("Selected-row comparison rule:")
    lines.append("- Always evaluate 'already included' relative to the existing final_choice rows from the original JSON.")
    lines.append("- Do NOT decide inclusion based on the statement in general; decide whether the D&A row appears already embedded in those selected final_choice row(s).")
    lines.append("- If a D&A row is labelled 'in COGS', 'in COR/COGS', 'in SGA', 'in R&D', or similar, treat that as evidence that it belongs to the same accounting bucket as the selected final_choice row(s),")
    lines.append("  but still determine whether it should be added separately by comparing it to those selected row(s) and their apparent hierarchical level.")
    lines.append("")

    lines.append('Reasoning guidance for "already included":')
    lines.append("- If explicit D&A rows exist as separate line items at the same hierarchical level as the selected COGS/XSGA rows,")
    lines.append("  and they are not indented beneath them and not labelled as sub-components of them,")
    lines.append("  assume D&A is NOT already included in the selected rows.")
    lines.append("- If no explicit D&A rows exist at that level, assume D&A IS already embedded in the selected rows.")
    lines.append("- If both a D&A summary/total row and its component rows exist, prefer the summary row to avoid double counting.")
    lines.append("- Only list component rows individually if no summary row exists.")
    lines.append("- A row is a D&A summary if its value equals or closely approximates the sum of the D&A component rows beneath it.")
    lines.append("")

    lines.append("Hierarchy interpretation guidance:")
    lines.append("- Use row ordering, wording, and nearby rows to infer hierarchy.")
    lines.append("- Rows such as 'Amortization in COR/COGS', 'Amortization in SGA', or 'Amortization in R&D' often indicate sub-components.")
    lines.append("- If a D&A row appears to be a separate expense line alongside the selected COGS/XSGA rows, treat it as NOT already included unless the statement strongly indicates otherwise.")
    lines.append("- If a D&A row clearly looks like a component nested under a selected parent/total row, do NOT add it separately.")
    lines.append("")

    lines.append("Parent-child inclusion rule:")
    lines.append("- If a selected final_choice row is clearly a total/parent row and a D&A row appears to be a component inside it, do NOT add that D&A row separately.")
    lines.append("- If the selected final_choice row is a component bucket that does NOT appear to include the D&A row, then you may add the D&A row separately.")
    lines.append("- If you are uncertain whether a separate D&A row should be added, set needs_manual_review=true.")
    lines.append("- In uncertain cases, use the most conservative output that avoids double counting.")
    lines.append("- That usually means final_choice=[] unless the statement clearly shows a separate same-level D&A row that should be added.")
    lines.append("")

    lines.append("Summary-vs-components rule:")
    lines.append("- If both a summary D&A row and more granular D&A rows are present for the same bucket, choose ONLY the summary row in final_choice.")
    lines.append("- Do not include both the summary row and its components.")
    lines.append("")

    lines.append("R&D interaction rule:")
    lines.append("- Earlier mapping used XSGA_COMPONENTS and XRD separately, and the PROF construction subtracts XRD from XSGA.")
    lines.append("- If an R&D expense row is present above operating profit, and that same R&D row is already included in BOTH XSGA_COMPONENTS final_choice and XRD final_choice, then do NOT add 'Amortization in R&D' or similar R&D D&A rows separately.")
    lines.append("- Reason: in that case, the same R&D amount is already included in SG&A and then subtracted again through XRD, so adding D&A in R&D separately would be unnecessary and may distort the logic.")
    lines.append("- If R&D is not visible as a separate row above operating profit in the provided income statement excerpt, but the original JSON indicates an XRD choice elsewhere, then an explicit row such as 'Amortization in R&D' may need to be added separately to XSGA_DA.")
    lines.append("- In that situation, if an explicit 'Amortization in R&D' row exists in the excerpt, you may include it in XSGA_DA final_choice if it appears to represent R&D-related D&A embedded in SG&A that would otherwise not be removed by the XRD subtraction.")
    lines.append("- For XSGA_DA, treat 'Amortization in R&D' as a special case: include it only when the separate above-operating-profit R&D row is not already being handled symmetrically through both XSGA_COMPONENTS and XRD.")
    lines.append("- If uncertain, prefer needs_manual_review=true and conservative output.")
    lines.append("")

    lines.append("Interpretation of final_choice:")
    lines.append("- final_choice must contain ONLY rows that should later be extracted and ADDED separately.")
    lines.append("- Therefore, if the visible D&A row is already embedded in the selected parent/total row, do not include it in final_choice.")
    lines.append("- If no separate D&A row should be added, use final_choice=[].")
    lines.append("- If both a summary D&A row and component D&A rows exist, use only the summary row in final_choice.")
    lines.append("")

    lines.append("Targets:")
    for k, desc in DA_TARGETS.items():
        lines.append(f"- {k}: {desc}")
    lines.append("")

    lines.append("Existing selected rows from the earlier PROF mapping:")
    for block in existing_choices["variables"]:
        lines.append(f"\n--- {block['variable']} already selected final_choice ---")
        if not block["final_choice"]:
            lines.append("(empty)")
        else:
            for ch in block["final_choice"]:
                lines.append(f"- [{ch['sheet_name']}] {ch['row_label']}")
        if block.get("notes"):
            lines.append(f"Notes: {block['notes']}")
    lines.append("")

    lines.append(f"=== SHEET: {income_sheet_name} (A–F preview, truncated above EBIT-like cutoff) ===")
    for r in income_preview[:250]:
        lines.append(" | ".join(r))
    if len(income_preview) > 250:
        lines.append(f"... ({len(income_preview) - 250} more rows omitted)")
    lines.append("")

    lines.append("Preferred candidate lists:")
    lines.append("- These candidate lists are only meant to focus attention.")
    lines.append("- You may choose rows outside the shortlist when appropriate.")
    for var, cands in shortlists.items():
        lines.append(f"\n--- CANDIDATES FOR {var} ---")
        if not cands:
            lines.append("(no candidates found)")
            continue
        for sheet, lab in cands:
            lines.append(f"- [{sheet}] {lab}")

    return "\n".join(lines)


# -----------------------------
# Responses API call
# -----------------------------
def _responses_create_da_json_schema(model: str, prompt: str, reasoning_effort: str) -> dict:
    """
    Calls the Responses API with structured JSON output.
    Tries two patterns for compatibility across SDK versions.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            response_format={"type": "json_schema", "json_schema": DA_MAPPING_SCHEMA},
        )
        if hasattr(resp, "output_parsed") and resp.output_parsed is not None:
            return resp.output_parsed
        return json.loads(resp.output_text)
    except TypeError:
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": DA_MAPPING_SCHEMA["name"],
                    "schema": DA_MAPPING_SCHEMA["schema"],
                    "strict": True,
                }
            },
        )
        return json.loads(resp.output_text)


# -----------------------------
# Validation
# -----------------------------
def validate_da_mapping_against_labels(mapping: dict, income_sheet_name: str, income_preview: list[list[str]]) -> dict:
    """
    Ensures every chosen row exists in the provided income statement excerpt.
    Invalid chosen rows are removed and the variable is flagged for manual review.
    """
    valid_labels = {r[0] for r in income_preview if r and r[0].strip() != ""}

    for v in mapping.get("variables", []):
        cleaned = []
        for ch in v.get("final_choice", []):
            s = ch.get("sheet_name", "")
            lab = ch.get("row_label", "")
            if s == income_sheet_name and lab in valid_labels:
                cleaned.append(ch)
            else:
                v["needs_manual_review"] = True
                msg = f"Chosen label not found in income statement excerpt: [{s}] {lab}"
                v["notes"] = (v.get("notes", "") + " | " + msg).strip(" |")
        v["final_choice"] = cleaned

    return mapping


# -----------------------------
# Main mapping function
# -----------------------------
def llm_map_da_rows(
    xlsx_path: Path,
    original_mapping_path: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict:
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    original_mapping = load_json(original_mapping_path)
    existing_choices = extract_existing_cogs_xsga_choices(original_mapping)

    income_sheet_name, income_preview = read_income_statement_excerpt_a_to_f(xlsx_path)

    shortlists: Dict[str, List[Tuple[str, str]]] = {}
    for var in DA_TARGETS.keys():
        shortlists[var] = shortlist_da_candidates(
            income_sheet_name=income_sheet_name,
            income_preview=income_preview,
            variable=var,
        )

    prompt = build_da_prompt(
        income_sheet_name=income_sheet_name,
        income_preview=income_preview,
        existing_choices=existing_choices,
        shortlists=shortlists,
    )

    mapping = _responses_create_da_json_schema(
        model=model,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
    )

    mapping = validate_da_mapping_against_labels(mapping, income_sheet_name, income_preview)

    return mapping


# -----------------------------
# Save helpers
# -----------------------------
def save_mapping(mapping: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Batch runner
# -----------------------------
def run_da_mapping_batch(
    input_dir: Path,
    original_mappings_dir: Path,
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_files = sorted(original_mappings_dir.glob("*.json"))
    print(f"Found {len(mapping_files)} original mapping files")

    n_ok, n_fail = 0, 0

    for mp in mapping_files:
        firm_id = mp.stem
        xlsx_path = input_dir / f"{firm_id}.xlsx"
        out_path = out_dir / f"{firm_id}.json"

        if not xlsx_path.exists():
            print(f"[SKIP] Missing xlsx for {firm_id}")
            n_fail += 1
            continue

        try:
            da_mapping = llm_map_da_rows(
                xlsx_path=xlsx_path,
                original_mapping_path=mp,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            save_mapping(da_mapping, out_path)
            print(f"[OK] {firm_id}")
            n_ok += 1

        except Exception as e:
            print(f"[FAIL] {firm_id}: {e}")
            n_fail += 1

    print(f"Done. OK={n_ok}, FAIL={n_fail}")


# -----------------------------
# Optional script entrypoint
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(".").resolve()

    INPUT_DIR = BASE_DIR / "data/stx_xlsx"
    ORIGINAL_MAPPINGS_DIR = BASE_DIR / "mappings/mappings_stx"
    OUT_DIR = BASE_DIR / "mappings/da_mappings_stx"

    run_da_mapping_batch(
        input_dir=INPUT_DIR,
        original_mappings_dir=ORIGINAL_MAPPINGS_DIR,
        out_dir=OUT_DIR,
        model=DEFAULT_MODEL,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
