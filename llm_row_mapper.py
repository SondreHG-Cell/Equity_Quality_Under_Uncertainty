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
DEFAULT_MODEL = "gpt-5.1"  # user request
# GPT-5.1 supports configurable reasoning effort via Responses API docs. :contentReference[oaicite:1]{index=1}
DEFAULT_REASONING_EFFORT = "medium"  # "none"|"low"|"medium"|"high" (try "low" first)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# -----------------------------
# PROF variables
# -----------------------------
# We map "XSGA" as COMPONENTS (often multiple rows)
PROF_TARGETS = {
    "REVT": "Total revenue / sales (top-line).",
    "COGS": "Cost of goods sold / cost of sales (direct production/service costs).",
    "XSGA_COMPONENTS": (
        "SG&A components (overhead). Often multiple rows such as Personnel/Administrative/Other operating expenses. "
        "May include stock-based compensation. EXCLUDE COGS, D&A, amortization, impairments."
    ),
    "XRD": "Research & development expense. Might be separate line, or embedded in SG&A.",
    "XINT": "Interest expense (prefer interest expense; net finance costs if that’s all there is).",
    "BE": "Book equity / total equity / shareholders' equity.",
    "MIB": "Minority interest / non-controlling interests.",
}

# Words that strongly indicate "totals" or non-usable balancing lines
FORBIDDEN_TOTAL_PATTERNS = [
    r"\btotal\b",
    r"\bsum\b",
    r"\bbalancing\b",
    r"\bremaining\b",
]


# -----------------------------
# Helpers: reading labels
# -----------------------------
def read_columns_a_to_f(
    xlsx_path: Path,
    max_rows: int = 5000,
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    """
    Reads columns A–F from each sheet, starting at the statement lines.
    Assumes the first real line items start at Excel row 19 across all sheets.

    Returns:
      labels_by_sheet: {sheet_name: [label1, label2, ...]}
      preview_by_sheet: {sheet_name: [[A,B,C,D,E,F], ...]} as strings
    """
    xl = pd.ExcelFile(xlsx_path)

    labels_by_sheet: Dict[str, List[str]] = {}
    preview_by_sheet: Dict[str, List[List[str]]] = {}

    # Excel row 19 is 1-indexed => skip first 18 rows
    SKIPROWS = 18
    sheet_names = xl.sheet_names[:2]
    for sheet in sheet_names:
        df = pd.read_excel(
            xlsx_path,
            sheet_name=sheet,
            usecols="A:F",
            skiprows=SKIPROWS,
            nrows=max_rows,
        )

        if df.shape[1] == 0:
            labels_by_sheet[sheet] = []
            preview_by_sheet[sheet] = []
            continue

        colA = df.columns[0]

        # Clean up
        df[colA] = df[colA].astype(str).str.strip()
        df = df.replace({pd.NA: "", "nan": "", "None": ""})

        # Build preview rows A–F as strings
        preview_rows: List[List[str]] = []
        for _, row in df.iterrows():
            row_vals = []
            for c in df.columns[:6]:
                v = row.get(c, "")
                v = "" if v is None else str(v).strip()
                row_vals.append(v)

            if row_vals[0] != "":
                preview_rows.append(row_vals)

        preview_by_sheet[sheet] = preview_rows

        # Build labels list (deduped, preserve order)
        labels = [r[0] for r in preview_rows if r[0].strip() != ""]
        seen = set()
        deduped = []
        for x in labels:
            if x not in seen:
                seen.add(x)
                deduped.append(x)

        labels_by_sheet[sheet] = deduped

    return labels_by_sheet, preview_by_sheet


# -----------------------------
# Candidate shortlisting (high impact)
# -----------------------------
@dataclass
class ShortlistRule:
    include: List[str]
    exclude: List[str]


SHORTLIST_RULES: Dict[str, ShortlistRule] = {
    "REVT": ShortlistRule(
        include=["revenue", "sales", "turnover", "driftsinntek", "omsetn"],
        exclude=["balancing", "other", "net", "income tax"],
    ),
    "COGS": ShortlistRule(
        include=[
            "cost of sales", "cogs", "cost of goods", "cost of services", "cost of revenue",
            "cost of materials", "materials", "raw materials", "consumables", "changes in inventory",
            "subcontract", "subcontractor", "external services",
            "direct costs", "project costs", "traffic charges",
            "purchases", "varekost", "vareforbruk", "material", "materialkost",
            "underentrepren", "direkte kost", "innkjøp"
        ],
        exclude=["total", "balancing", "depreciation", "amortization", "impairment", "interest", "tax", "finance"],
    ),
    "XSGA_COMPONENTS": ShortlistRule(
        include=[
            "selling, general", "selling general", "sg&a", "sga", "sales and administration",
            "personnel", "salary", "wages", "lønn", "lonn", "administrative", "admin",
            "selling", "general", "g&a", "g & a", "other operating", "andre drifts",
            "operating expenses", "stock options", "share-based", "equity compensation"
        ],
        exclude=[
            "total", "balancing", "cost of sales", "cogs", "depreciation", "amortization",
            "impairment", "interest", "tax", "finance", "financial"
        ],
    ),
    "XRD": ShortlistRule(
        include=["r&d", "research", "development", "forskning", "utvikling"],
        exclude=["total", "balancing"],
    ),
    "XINT": ShortlistRule(
        include=["interest expense", "interest", "finanskost", "rentekost", "net finance"],
        exclude=["total", "balancing", "interest income", "financial income"],
    ),
    "BE": ShortlistRule(
        include=["total equity", "equity", "egenkap", "shareholders' equity", "owners' equity"],
        exclude=["total assets", "liabilities", "balancing"],
    ),
    "MIB": ShortlistRule(
        include=["minority", "non-controlling", "noncontrolling", "minoritets", "interests"],
        exclude=["total", "balancing"],
    ),
}


def _contains_any(s: str, needles: List[str]) -> bool:
    s2 = s.lower()
    return any(n.lower() in s2 for n in needles)


def _contains_forbidden_total(s: str) -> bool:
    s2 = s.lower()
    return any(re.search(p, s2) for p in FORBIDDEN_TOTAL_PATTERNS)


def shortlist_candidates(sheet_labels: Dict[str, List[str]], variable: str, max_per_sheet: int = 30) -> List[Tuple[str, str]]:
    """
    Returns list of (sheet_name, row_label) candidates.
    We filter labels using keyword rules to keep the LLM focused and avoid "Total operating expenses".
    """
    rule = SHORTLIST_RULES[variable]
    out: List[Tuple[str, str]] = []

    for sheet, labels in sheet_labels.items():
        picks = []
        for lab in labels:
            if _contains_forbidden_total(lab):
                # For most variables, totals are suspicious; for REVT/BE totals can be ok,
                # but we still don't want generic "Total operating expenses" for XSGA.
                if variable == "XSGA_COMPONENTS":
                    continue

            lo = lab.lower()
            if _contains_any(lo, rule.include) and not _contains_any(lo, rule.exclude):
                picks.append(lab)

        # cap per sheet
        for lab in picks[:max_per_sheet]:
            out.append((sheet, lab))

    # Fallback: if shortlist too small, allow a broader pass (still avoiding totals for XSGA)
    if len(out) < 5:
        for sheet, labels in sheet_labels.items():
            for lab in labels:
                lo = lab.lower()
                if variable == "XSGA_COMPONENTS" and _contains_forbidden_total(lab):
                    continue
                if _contains_any(lo, rule.include):
                    out.append((sheet, lab))
            if len(out) >= 30:
                break
    # Deduplicate (sheet, label) while preserving order
    seen = set()
    deduped = []
    for sheet, lab in out:
        key = (sheet, lab)
        if key not in seen:
            seen.add(key)
            deduped.append((sheet, lab))

    return deduped[:80]


# -----------------------------
# JSON schema for output
#   - final_choice is ALWAYS a list (supports multi-row XSGA)
#   - if uncertain, set needs_manual_review=true and confidence low
# -----------------------------
MAPPING_SCHEMA = {
    "name": "prof_row_mapping_v2",
    "schema": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "enum": list(PROF_TARGETS.keys())},
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


def build_prompt(
    sheet_labels: dict[str, list[str]],
    sheet_preview: dict[str, list[list[str]]],
    shortlists: dict[str, list[tuple[str, str]]],
) -> str:
    """
    Builds an instruction prompt for the LLM.

    Inputs:
      - sheet_labels: {sheet_name: [label1, label2, ...]} from column A (deduped)
      - sheet_preview: {sheet_name: [[A,B,C,D,E,F], ...]} first 6 columns as strings
      - shortlists: {variable: [(sheet_name, row_label), ...]} candidates per variable

    Output:
      - prompt string (LLM should respond with JSON matching your schema)
    """
    lines: list[str] = []

    lines.append("You are mapping accounting statement row labels to PROF inputs.")
    lines.append("You MUST choose row labels EXACTLY as written in column A.")
    lines.append("You MUST choose only from the provided candidate lists for each variable.")
    lines.append("Return JSON ONLY that conforms to the given schema.")
    lines.append("")

    lines.append("Targets:")
    for k, desc in PROF_TARGETS.items():
        lines.append(f"- {k}: {desc}")
    lines.append("")

    lines.append("Critical rules:")
    lines.append("1) Choose row labels EXACTLY as written (case/punctuation must match).")
    lines.append("2) Do NOT invent labels that are not present.")
    lines.append("3) Prefer component rows over subtotal/total rows.")
    lines.append("4) Exclude rows that are clearly balancing/subtotal/total when component rows exist.")
    lines.append("")

    # XSGA-specific rules (the main pain point)
    lines.append("Critical rules for XSGA_COMPONENTS (SG&A):")
    lines.append("- XSGA_COMPONENTS is operating overhead and is often MULTIPLE rows (choose several if needed).")
    lines.append("- NEVER select totals/subtotals such as 'Total operating expenses', 'Total ...', 'Balancing Item ...', or '... Remaining'.")
    lines.append("- NEVER select rows containing these phrases for XSGA_COMPONENTS: Total, Sum, Balancing, Remaining.")
    lines.append("- Prefer overhead components like: Personnel costs, Sales and administration costs, Administrative expenses, Selling expenses, Other operating expenses, Cost of Stock Options / share-based compensation.")
    lines.append("- EXCLUDE: Cost of sales/COGS, depreciation, amortization, impairments, interest, taxes, finance items.")
    lines.append(
        "- XSGA selection hierarchy:"
    )
    lines.append(
        "  (a) If an explicit SG&A row exists (e.g., 'Selling, General & Admin' / 'Sales and administration costs') "
        "AND it appears populated in the A–F preview (i.e., has non-empty numbers in columns B–F), "
        "then select ONLY that row for XSGA_COMPONENTS."
    )
    lines.append(
        "  (b) If an explicit SG&A row exists but appears mostly empty in the A–F preview, "
        "then include that SG&A row AND ALSO include 'Operating expenses' (or the closest overhead bucket such as "
        "'Other operating expenses' / 'Personnel costs') to ensure coverage for earlier years."
    )
    lines.append(
        "  (c) If no explicit SG&A row exists, select the best overhead bucket(s) such as "
        "'Operating expenses', 'Other operating expenses', 'Personnel costs', and similar."
)
    lines.append("")

    lines.append("Missing data rule (IMPORTANT):")
    lines.append("- If a target variable truly does not exist in the statements, set final_choice to [].")
    lines.append("- This will be interpreted as 0 later when computing PROF (e.g., XRD is often missing).")
    lines.append("- Set needs_manual_review=true ONLY when you suspect the item exists but you cannot confidently identify it.")
    lines.append("- If it does not exist, use final_choice=[] and needs_manual_review=false.")
    lines.append("")

    lines.append("IMPORTANT DATA RULE:")
    lines.append("- If a row label is present in the sheet, it means it has a value in at least one year (even if columns B–F are empty in this preview).")
    lines.append("- Therefore, do NOT exclude a row just because its A–F preview values are empty or 'nan'.")
    lines.append("- Prefer the most semantically correct label (e.g., an explicit 'Selling, General & Admin' row) even if its preview values appear empty.")
    lines.append("")

    # Provide A–F preview (this helps model avoid sums-of-sums / duplicate totals)
    lines.append("Context: Below is a preview of the first 6 columns (A–F) from each sheet.")
    lines.append("Column A is the row label. Columns B–F are recent values/years when present.")
    lines.append("Use this numeric preview to detect duplicated totals/subtotals (e.g., a 'Total ...' row identical to a component row).")
    lines.append("")

    for sheet, rows in sheet_preview.items():
        lines.append(f"\n=== SHEET: {sheet} (A–F preview) ===")
        # cap to keep prompt size under control
        for r in rows[:200]:
            # r is [A, B, C, D, E, F] (some may be empty)
            lines.append(" | ".join(r))
        if len(rows) > 200:
            lines.append(f"... ({len(rows) - 200} more rows omitted)")
    lines.append("")

    # Candidate lists (hard constraint for the model)
    lines.append("Now choose ONLY from the candidate lists below for each variable.")
    lines.append("Do not choose any label outside these lists.")
    for var, cands in shortlists.items():
        lines.append(f"\n--- CANDIDATES FOR {var} ---")
        if not cands:
            lines.append("(no candidates found)")
            continue
        for sheet, lab in cands:
            lines.append(f"- [{sheet}] {lab}")

    return "\n".join(lines)


def _responses_create_json_schema(model: str, prompt: str, reasoning_effort: str) -> dict:
    """
    Calls the Responses API with JSON schema output. Different SDK versions support
    slightly different parameter names, so we try a couple of compatible patterns.
    """
    # Pattern A: response_format (some SDKs)
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            response_format={"type": "json_schema", "json_schema": MAPPING_SCHEMA},
        )
        # Some SDKs provide output_parsed
        if hasattr(resp, "output_parsed") and resp.output_parsed is not None:
            return resp.output_parsed
        return json.loads(resp.output_text)
    except TypeError:
        # Pattern B: text.format (documented in structured outputs guide) :contentReference[oaicite:2]{index=2}
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": MAPPING_SCHEMA["name"],
                    "schema": MAPPING_SCHEMA["schema"],
                    "strict": True,
                }
            },
        )
        return json.loads(resp.output_text)


def llm_map_prof_rows(
    xlsx_path: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict:
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    sheet_labels, sheet_preview = read_columns_a_to_f(xlsx_path)

    shortlists: Dict[str, List[Tuple[str, str]]] = {}
    for var in PROF_TARGETS.keys():
        shortlists[var] = shortlist_candidates(sheet_labels, var)

    # FIX: pass sheet_preview too
    prompt = build_prompt(sheet_labels, sheet_preview, shortlists)

    return _responses_create_json_schema(model=model, prompt=prompt, reasoning_effort=reasoning_effort)


def save_mapping(mapping: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")