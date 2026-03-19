from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
load_dotenv()

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"  # "none"|"low"|"medium"|"high"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# -----------------------------
# Accruals variables
# -----------------------------
ACCRUALS_TARGETS = {
    "ACT": "Total current assets.",
    "CHE": "Cash and cash equivalents.",
    "LCT": "Total current liabilities.",
    "DLC": "Short-term debt or current portion of long-term debt.",
    "TXP": "Taxes payable / current tax liabilities / income taxes payable.",
    "OANCF": "Net cash flow from operating activities.",
    "REVT": "Total revenue / sales / operating revenue.",
    "PPEGT": "Property, plant and equipment (prefer total PPE / net PPE line).",
    "AT": "Total assets.",
}

# These patterns often indicate generic balancing rows or broad subtotals that are not useful.
FORBIDDEN_TOTAL_PATTERNS = [
    r"\bsum\b",
    r"\bbalancing\b",
    r"\bremaining\b",
]

# These variables normally should map to a single reported line.
SINGLE_LINE_TARGETS = {"ACT", "CHE", "LCT", "DLC", "TXP", "OANCF", "PPEGT", "AT"}

# These variables may reasonably change label over time, so multiple priority labels can be returned.
PRIORITY_LIST_TARGETS = {"REVT"}


# -----------------------------
# Helpers: reading labels
# -----------------------------
def read_columns_a_to_f(
    xlsx_path: Path,
    max_rows: int = 5000,
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    """
    Read columns A-F from each sheet, starting near the statement rows.

    Returns:
      labels_by_sheet: {sheet_name: [label1, label2, ...]}
      preview_by_sheet: {sheet_name: [[A,B,C,D,E,F], ...]} as strings
    """
    xl = pd.ExcelFile(xlsx_path)

    labels_by_sheet: Dict[str, List[str]] = {}
    preview_by_sheet: Dict[str, List[List[str]]] = {}

    # The original workbook structure starts statement rows around Excel row 19.
    skiprows = 17

    for sheet in xl.sheet_names[:3]:
        df = pd.read_excel(
            xlsx_path,
            sheet_name=sheet,
            usecols="A:F",
            skiprows=skiprows,
            nrows=max_rows,
        )

        if df.shape[1] == 0:
            labels_by_sheet[sheet] = []
            preview_by_sheet[sheet] = []
            continue

        col_a = df.columns[0]
        df[col_a] = df[col_a].astype(str).str.strip()
        df = df.replace({pd.NA: "", "nan": "", "None": ""})

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

        labels = [r[0] for r in preview_rows if r[0].strip() != ""]
        seen = set()
        deduped = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                deduped.append(label)

        labels_by_sheet[sheet] = deduped

    return labels_by_sheet, preview_by_sheet


# -----------------------------
# Candidate shortlisting
# -----------------------------
@dataclass
class ShortlistRule:
    include: List[str]
    exclude: List[str]
    preferred_sheets: List[str] = field(default_factory=list)
    allow_totals: bool = True
    broad_fallback: bool = True


SHORTLIST_RULES = {
    "ACT": ShortlistRule(
        include=[
            "total current assets", "current assets", "curr assets", "current asset",
        ],
        exclude=["non-current", "noncurrent", "total assets", "assets held for sale"],
        preferred_sheets=["balance sheet", "statement of financial position", "financial position", "balance"],
        allow_totals=True,
        broad_fallback=True,
    ),
    "CHE": ShortlistRule(
        include=[
            "cash and cash equivalents", "cash & cash equivalents", "cash and equivalents",
            "cash equivalents", "cash at bank", "cash",
        ],
        exclude=[
            "restricted cash", "cash flow", "change in cash", "net cash", "cash generated",
            "interest paid", "interest received", "exchange rate difference", "money market",
            "marketable securities", "market securities", "receivable", "trade receivable",
        ],
        preferred_sheets=["balance sheet", "statement of financial position", "financial position", "balance"],
        allow_totals=False,
        broad_fallback=True,
    ),
    "LCT": ShortlistRule(
        include=["total current liabilities", "current liabilities", "curr liabilities", "current liability"],
        exclude=["non-current", "noncurrent", "total liabilities and equity", "equity"],
        preferred_sheets=["balance sheet", "statement of financial position", "financial position", "balance"],
        allow_totals=True,
        broad_fallback=True,
    ),
    "DLC": ShortlistRule(
        include=[
            "short-term debt", "short term debt", "short-term borrowings", "short term borrowings",
            "current portion of long-term debt", "current portion of long term debt",
            "current portion of debt", "current debt", "interest-bearing liabilities, current",
            "lease liabilities, current", "current lease liabilities", "current borrowings",
        ],
        exclude=[
            "non-current", "noncurrent", "total liabilities", "receivable", "cash", "accounts payable",
            "trade payables", "other payables", "accrued liabilities", "tax payable", "current liabilities",
        ],
        preferred_sheets=["balance sheet", "statement of financial position", "financial position", "balance"],
        allow_totals=False,
        broad_fallback=True,
    ),
    "TXP": ShortlistRule(
        include=[
            "tax payable", "taxes payable", "income tax payable", "income taxes payable",
            "income taxes - payable", "current tax liability", "current tax liabilities",
            "current tax payable", "current tax payables", "income taxes - payable - short-term",
        ],
        exclude=[
            "deferred", "deferred tax", "deferred taxes", "deferred tax liabilities",
            "income tax expense", "tax expense", "tax expenses", "current tax expenses",
            "income taxes - total", "tax receivable", "tax asset", "indirect taxes",
            "public charges", "adjustment", "balancing", "profit before tax", "after tax",
            "net of tax", "provision",
        ],
        preferred_sheets=["balance sheet", "statement of financial position", "financial position", "balance"],
        allow_totals=False,
        broad_fallback=False,
    ),
    "OANCF": ShortlistRule(
        include=[
            "cash flow from operating", "net cash flow from operating activities",
            "net cash from operating", "net cash generated from operating",
            "operating activities", "operating cash flow", "cash generated from operations",
        ],
        exclude=["investing", "financing", "change in cash", "free cash flow"],
        preferred_sheets=["cash flow", "cash flows", "statement of cash flow", "statement of cash flows"],
        allow_totals=True,
        broad_fallback=True,
    ),
    "REVT": ShortlistRule(
        include=[
            "total revenue", "revenue", "revenues", "sales", "turnover",
            "operating revenue", "operating revenues", "net sales",
        ],
        exclude=[
            "other", "interest income", "financial income", "net income",
            "operating income", "tax", "comprehensive income",
        ],
        preferred_sheets=["income statement", "statement of profit", "statement of income", "profit and loss", "p&l"],
        allow_totals=True,
        broad_fallback=True,
    ),
    "PPEGT": ShortlistRule(
        include=[
            "property, plant and equipment", "property plant and equipment",
            "property, plant", "property plant", "ppe", "plant and equipment",
            "tangible fixed assets", "tangible assets",
        ],
        exclude=[
            "right-of-use", "right of use", "intangible", "goodwill", "investment property",
            "depreciation", "accumulated depreciation", "machinery and equipment under construction",
        ],
        preferred_sheets=["balance sheet", "statement of financial position", "financial position", "balance"],
        allow_totals=True,
        broad_fallback=True,
    ),
    "AT": ShortlistRule(
        include=["total assets", "assets - total", "assets, total"],
        exclude=["current assets", "non-current assets", "noncurrent assets"],
        preferred_sheets=["balance sheet", "statement of financial position", "financial position", "balance"],
        allow_totals=True,
        broad_fallback=True,
    ),
}


def _contains_any(text: str, needles: List[str]) -> bool:
    text = text.lower()
    return any(needle.lower() in text for needle in needles)


def _contains_forbidden_total(text: str) -> bool:
    text = text.lower()
    return any(re.search(pattern, text) for pattern in FORBIDDEN_TOTAL_PATTERNS)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _sheet_matches_preferences(sheet_name: str, preferred_sheets: List[str]) -> bool:
    if not preferred_sheets:
        return True
    normalized_sheet = _normalize_text(sheet_name)
    return any(token in normalized_sheet for token in preferred_sheets)


def _score_candidate_label(label: str, variable: str, is_preferred_sheet: bool) -> int:
    label_lower = _normalize_text(label)
    score = 0

    if is_preferred_sheet:
        score += 100

    if variable in {"ACT", "LCT", "AT", "REVT", "OANCF"} and "total" in label_lower:
        score += 20

    if variable == "CHE":
        if "cash and cash equivalents" in label_lower or "cash & cash equivalents" in label_lower:
            score += 35
        elif "cash equivalents" in label_lower:
            score += 15
        elif label_lower == "cash":
            score += 5

    if variable == "DLC":
        if "current portion" in label_lower:
            score += 25
        if "short-term" in label_lower or "short term" in label_lower:
            score += 20
        if "lease" in label_lower:
            score += 8

    if variable == "TXP":
        if "payable" in label_lower:
            score += 30
        if "current tax" in label_lower:
            score += 25
        if "income tax" in label_lower or "income taxes" in label_lower:
            score += 10

    if variable == "PPEGT":
        if "property, plant and equipment" in label_lower or "property plant and equipment" in label_lower:
            score += 35
        elif label_lower == "ppe":
            score += 25
        elif "tangible fixed assets" in label_lower or "tangible assets" in label_lower:
            score += 10

    if variable == "REVT":
        if "net sales" in label_lower or "turnover" in label_lower:
            score += 10

    return score


def shortlist_candidates(sheet_labels: Dict[str, List[str]], variable: str, max_per_sheet: int = 30) -> List[Tuple[str, str]]:
    """
    Return a focused list of (sheet_name, row_label) candidates for a variable.
    The goal is to keep the model focused on plausible rows while respecting statement type.
    """
    rule = SHORTLIST_RULES[variable]
    scored_candidates: List[Tuple[int, str, str]] = []

    for sheet, labels in sheet_labels.items():
        is_preferred_sheet = _sheet_matches_preferences(sheet, rule.preferred_sheets)

        for label in labels:
            label_lower = _normalize_text(label)

            if _contains_forbidden_total(label) and not rule.allow_totals:
                continue

            if _contains_any(label_lower, rule.include) and not _contains_any(label_lower, rule.exclude):
                score = _score_candidate_label(label, variable, is_preferred_sheet)
                scored_candidates.append((score, sheet, label))

    scored_candidates.sort(key=lambda x: (-x[0], x[1].lower(), x[2].lower()))

    per_sheet_counts: Dict[str, int] = {}
    out: List[Tuple[str, str]] = []
    for _, sheet, label in scored_candidates:
        count = per_sheet_counts.get(sheet, 0)
        if count >= max_per_sheet:
            continue
        out.append((sheet, label))
        per_sheet_counts[sheet] = count + 1

    if len(out) < 5 and rule.broad_fallback:
        fallback_candidates: List[Tuple[int, str, str]] = []
        for sheet, labels in sheet_labels.items():
            is_preferred_sheet = _sheet_matches_preferences(sheet, rule.preferred_sheets)
            for label in labels:
                label_lower = _normalize_text(label)
                if _contains_any(label_lower, rule.include):
                    score = _score_candidate_label(label, variable, is_preferred_sheet) - 15
                    fallback_candidates.append((score, sheet, label))

        fallback_candidates.sort(key=lambda x: (-x[0], x[1].lower(), x[2].lower()))
        for _, sheet, label in fallback_candidates:
            out.append((sheet, label))
            if len(out) >= 30:
                break

    seen = set()
    deduped = []
    for sheet, label in out:
        key = (sheet, label)
        if key not in seen:
            seen.add(key)
            deduped.append((sheet, label))

    return deduped[:80]


# -----------------------------
# JSON schema for output
# -----------------------------
MAPPING_SCHEMA = {
    "name": "accruals_row_mapping_v1",
    "schema": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "enum": list(ACCRUALS_TARGETS.keys())},
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
# Prompt builder
# -----------------------------
def build_prompt(
    sheet_labels: dict[str, list[str]],
    sheet_preview: dict[str, list[list[str]]],
    shortlists: dict[str, list[tuple[str, str]]],
) -> str:
    """
    Build a prompt for accruals row mapping.
    The model should prefer the shortlisted candidates, but may use the full preview as an escape hatch.
    """
    lines: list[str] = []

    lines.append("You are mapping accounting statement row labels to accruals-model inputs.")
    lines.append("You MUST choose row labels EXACTLY as written in column A.")
    lines.append("You should choose from the provided candidate lists whenever possible.")
    lines.append("Use sheet context aggressively: Balance Sheet items should usually come from balance-sheet-type sheets, OANCF from cash-flow-type sheets, and REVT from income-statement-type sheets.")
    lines.append(
        "If a candidate list is empty OR clearly misses the correct label, you may choose a row_label from the full A-column labels shown in the A-F preview. "
        "In that case you MUST set needs_manual_review=true and explain why in notes."
    )
    lines.append("Return JSON ONLY that conforms to the given schema.")
    lines.append("")

    lines.append("Targets:")
    for key, desc in ACCRUALS_TARGETS.items():
        lines.append(f"- {key}: {desc}")
    lines.append("")

    lines.append("General rules:")
    lines.append("1) Choose row labels EXACTLY as written. Do not invent labels.")
    lines.append("2) Prefer reported total lines over components whenever a clear total line exists.")
    lines.append("3) Do NOT choose a component row if a better total line exists for the same concept.")
    lines.append("4) Do NOT repeat the same row_label twice in final_choice.")
    lines.append("5) Use final_choice=[] only if the target truly does not exist or cannot be identified.")
    lines.append(
        "6) You may choose outside the candidate list only when the shortlist is empty or clearly wrong. "
        "If you do, set needs_manual_review=true."
    )
    lines.append("")

    lines.append("Variable-specific rules:")
    lines.append("- ACT: Prefer 'Total current assets' or the closest equivalent total current assets line.")
    lines.append("- CHE: Prefer pure cash and cash equivalents on the balance sheet. Avoid restricted cash, cash flow rows, receivables, and non-cash liquidity items.")
    lines.append("- LCT: Prefer 'Total current liabilities' or the closest equivalent total current liabilities line.")
    lines.append("- DLC: Prefer explicit short-term debt or current portion of long-term debt on the balance sheet. Avoid total liabilities, generic payables, and broad current-liability rows.")
    lines.append("- TXP: Prefer explicit taxes payable / current tax liabilities on the balance sheet. Avoid deferred tax, provisions, tax assets, and tax expense rows.")
    lines.append("- OANCF: Prefer the total operating cash flow line from the cash flow statement.")
    lines.append("- REVT: Prefer the total revenue line from the income statement. Avoid operating income, net income, and tax rows. REVT may use multiple labels over time, so you may return a priority list.")
    lines.append("- PPEGT: Prefer the main PPE line (total PPE / net PPE) on the balance sheet. Tangible fixed assets are an acceptable fallback. Avoid goodwill, intangibles, and right-of-use assets if possible.")
    lines.append("- AT: Prefer the reported total assets line.")
    lines.append("")

    lines.append("Missing-data rule:")
    lines.append("- If a target variable truly does not exist in the workbook, use final_choice=[].")
    lines.append("- Set needs_manual_review=true only if you think the item exists but you cannot identify it confidently.")
    lines.append("")

    lines.append("Selection-size rule:")
    lines.append("- For ACT, CHE, LCT, DLC, TXP, OANCF, PPEGT, and AT, final_choice should usually contain exactly one row.")
    lines.append("- For TXP in particular, it is better to return final_choice=[] than to choose a likely tax-expense or deferred-tax row.")
    lines.append("- For REVT, final_choice may contain multiple rows as a priority list if the label changes over time.")
    lines.append("")

    lines.append("IMPORTANT DATA RULE:")
    lines.append("- If a row label is present in the sheet, it may still be valid even if columns B-F are empty in the preview.")
    lines.append("- Therefore, do NOT reject a semantically correct row only because recent preview values are blank.")
    lines.append("")

    lines.append("Context: Below is a preview of the first 6 columns (A-F) from each sheet.")
    lines.append("Column A is the row label. Columns B-F are recent values.")
    lines.append("")

    for sheet, rows in sheet_preview.items():
        lines.append(f"\n=== SHEET: {sheet} (A-F preview) ===")
        for row in rows[:200]:
            lines.append(" | ".join(row))
        if len(rows) > 200:
            lines.append(f"... ({len(rows) - 200} more rows omitted)")
    lines.append("")

    lines.append("Preferred candidate lists (use these whenever possible):")
    for variable, cands in shortlists.items():
        lines.append(f"\n--- CANDIDATES FOR {variable} ---")
        if not cands:
            lines.append("(no candidates found)")
            continue
        for sheet, label in cands:
            lines.append(f"- [{sheet}] {label}")

    return "\n".join(lines)


# -----------------------------
# API call helper
# -----------------------------
def _responses_create_json_schema(model: str, prompt: str, reasoning_effort: str) -> dict:
    """
    Call the Responses API with structured JSON output.
    Some SDK versions use different parameter names, so we try two compatible patterns.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            response_format={"type": "json_schema", "json_schema": MAPPING_SCHEMA},
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
                    "name": MAPPING_SCHEMA["name"],
                    "schema": MAPPING_SCHEMA["schema"],
                    "strict": True,
                }
            },
        )
        return json.loads(resp.output_text)


# -----------------------------
# Validation and post-processing
# -----------------------------
def validate_mapping_against_labels(mapping: dict, sheet_labels: dict[str, list[str]]) -> dict:
    """
    Ensure every (sheet_name, row_label) in final_choice exists in the workbook labels.
    Invalid choices are removed and the variable is flagged for manual review.
    """
    labels_set = {sheet: set(labels) for sheet, labels in sheet_labels.items()}

    for variable in mapping.get("variables", []):
        cleaned = []
        for choice in variable.get("final_choice", []):
            sheet = choice.get("sheet_name", "")
            label = choice.get("row_label", "")
            if sheet in labels_set and label in labels_set[sheet]:
                cleaned.append(choice)
            else:
                variable["needs_manual_review"] = True
                msg = f"Chosen label not found in sheet: [{sheet}] {label}"
                variable["notes"] = (variable.get("notes", "") + " | " + msg).strip(" |")
        variable["final_choice"] = cleaned

    return mapping


def flag_escape_hatch_choices(mapping: dict, shortlists: dict[str, list[tuple[str, str]]]) -> dict:
    """
    Flag manual review if the model chooses a final row outside the candidate shortlist.
    """
    shortlist_sets = {
        variable: set((sheet, label) for (sheet, label) in cands)
        for variable, cands in shortlists.items()
    }

    for variable in mapping.get("variables", []):
        var_name = variable.get("variable", "")
        allowed = shortlist_sets.get(var_name, set())
        finals = variable.get("final_choice", [])

        if not finals:
            continue

        outside = []
        for choice in finals:
            key = (choice.get("sheet_name", ""), choice.get("row_label", ""))
            if allowed and key not in allowed:
                outside.append(key)

        if outside:
            variable["needs_manual_review"] = True
            msg = "Escape-hatch used (final_choice outside candidate list): " + ", ".join(
                [f"[{sheet}] {label}" for sheet, label in outside]
            )
            variable["notes"] = (variable.get("notes", "") + " | " + msg).strip(" |")

    return mapping


def enforce_target_shape(mapping: dict) -> dict:
    """
    Enforce simple shape rules after the model returns its output.

    - Single-line targets are reduced to the first selected row if the model returns more than one.
    - REVT may keep multiple labels as a priority list.
    """
    for variable in mapping.get("variables", []):
        var_name = variable.get("variable", "")
        finals = variable.get("final_choice", [])

        if var_name in SINGLE_LINE_TARGETS and len(finals) > 1:
            variable["final_choice"] = finals[:1]
            variable["needs_manual_review"] = True
            msg = "Reduced multi-row final_choice to first row for single-line target."
            variable["notes"] = (variable.get("notes", "") + " | " + msg).strip(" |")

    return mapping


# -----------------------------
# Main public functions
# -----------------------------
def llm_map_accruals_rows(
    xlsx_path: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict:
    """
    Map workbook rows to accruals-model variables.
    """
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    sheet_labels, sheet_preview = read_columns_a_to_f(xlsx_path)

    shortlists: Dict[str, List[Tuple[str, str]]] = {}
    for variable in ACCRUALS_TARGETS.keys():
        shortlists[variable] = shortlist_candidates(sheet_labels, variable)

    prompt = build_prompt(sheet_labels, sheet_preview, shortlists)
    mapping = _responses_create_json_schema(model=model, prompt=prompt, reasoning_effort=reasoning_effort)

    mapping = flag_escape_hatch_choices(mapping, shortlists)
    mapping = validate_mapping_against_labels(mapping, sheet_labels)
    mapping = enforce_target_shape(mapping)

    return mapping


def save_mapping_accruals(mapping: dict, out_path: Path) -> None:
    """
    Save an accruals mapping JSON file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
