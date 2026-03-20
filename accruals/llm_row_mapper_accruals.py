from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from openai import OpenAI


# =============================================================================
# Config
# =============================================================================
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "low"  # "none"|"minimal"|"low"|"medium"|"high"|"xhigh"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Baseline choice for Nordic/IFRS: exclude leases from STD (short-term debt)
INCLUDE_LEASES_IN_STD = False  # set True for robustness version

# Confidence threshold for auto-flagging manual review
CONFIDENCE_THRESHOLD = 0.65


# =============================================================================
# Targets
# =============================================================================
# Working capital accruals need: ΔCA, ΔCash, ΔCL, ΔSTD, ΔTP
# Revenue is handled elsewhere (from your PROF extraction), so we do NOT map REVT here.
#
# Variable naming:
# - We use "STD" to align with the typical accruals formula.
ACCRUALS_TARGETS = {
    "ACT": "Total current assets (balance sheet).",
    "CHE": "Cash and cash equivalents (balance sheet).",
    "LCT": "Total current liabilities (balance sheet).",
    "STD": "Short-term debt / current portion of long-term debt (balance sheet). Exclude leases in baseline.",
    "TXP": "Taxes payable / current tax liabilities payable (balance sheet).",
    "PPEGT": "Property, plant and equipment (balance sheet; prefer total/net PPE).",
    "AT": "Total assets (balance sheet).",
    "OANCF": "Net cash flow from operating activities / operating cash flow (cash flow statement).",
}
ALL_VARS = list(ACCRUALS_TARGETS.keys())

# Variables interpreted as priority lists later (first non-missing per year)
PRIORITY_LIST_VARS = {"ACT", "CHE", "LCT", "TXP", "PPEGT", "AT", "OANCF"}

# Variables interpreted as sums later (sum of components per year)
SUM_VARS = {"STD"}  # under IFRS, short-term debt is often split across multiple lines

# Light “balancing” patterns (avoid only when clearly a balancing line)
FORBIDDEN_BALANCING_PATTERNS = [
    r"\bbalancing\b",
    r"\bremaining\b",
    r"\bsubtotal\b",
]


# =============================================================================
# Sheet reading helpers (Field Name header detection)
# =============================================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def detect_field_name_header_row(
    xlsx_path: Path,
    sheet_name: str,
    max_scan_rows: int = 140,
) -> Optional[int]:
    """
    Find 0-indexed row where column A equals 'Field Name'.
    Reads only column A to avoid out-of-bounds issues.
    """
    scan = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        header=None,
        nrows=max_scan_rows,
        usecols="A",
        engine="openpyxl",
    )

    for i in range(len(scan)):
        v = scan.iloc[i, 0]
        v = "" if pd.isna(v) else str(v).strip()
        if v.lower() == "field name":
            return i
    return None


def classify_sheet_type(sheet_name: str) -> str:
    s = _norm(sheet_name)
    if any(tok in s for tok in ["balance", "financial position", "statement of financial position"]):
        return "balance"
    if any(tok in s for tok in ["cash flow", "cashflow", "statement of cash flows", "statement of cash flow"]):
        return "cashflow"
    return "other"


def read_sheet_a_to_f_preview(
    xlsx_path: Path,
    sheet_name: str,
    max_rows_after_header: int = 5000,
) -> Tuple[List[str], List[List[str]]]:
    """
    Read A–F preview using 'Field Name' as header when available.
    Returns:
      labels: deduped labels from col A (order preserved)
      preview_rows: rows [A,B,C,D,E,F] as strings
    """
    header_row = detect_field_name_header_row(xlsx_path, sheet_name)
    if header_row is None:
        # Fallback for rare files where detection fails
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, usecols="A:F", skiprows=18)
    else:
        df = pd.read_excel(
            xlsx_path,
            sheet_name=sheet_name,
            header=header_row,
            usecols="A:F",
            nrows=max_rows_after_header,
        )

    if df.shape[1] == 0:
        return [], []

    colA = df.columns[0]
    df[colA] = df[colA].astype(str).str.strip()
    df = df.replace({pd.NA: "", "nan": "", "None": ""})

    preview_rows: List[List[str]] = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns[:6]:
            v = row.get(c, "")
            v = "" if v is None else str(v).strip()
            vals.append(v)
        if vals[0] != "":
            preview_rows.append(vals)

    # dedupe labels
    labels = [r[0] for r in preview_rows if str(r[0]).strip() != ""]
    seen = set()
    deduped = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            deduped.append(lab)

    return deduped, preview_rows


def read_columns_a_to_f_balance_and_cashflow(
    xlsx_path: Path,
    max_rows_after_header: int = 5000,
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    """
    Only include Balance Sheet-type and Cash Flow-type sheets (by name).
    If none found, fall back to first 3 sheets.
    """
    xl = pd.ExcelFile(xlsx_path)

    selected = []
    for sh in xl.sheet_names:
        if classify_sheet_type(sh) in {"balance", "cashflow"}:
            selected.append(sh)

    if not selected:
        selected = xl.sheet_names[:3]

    labels_by_sheet: Dict[str, List[str]] = {}
    preview_by_sheet: Dict[str, List[List[str]]] = {}
    for sh in selected:
        labels, prev = read_sheet_a_to_f_preview(xlsx_path, sh, max_rows_after_header=max_rows_after_header)
        labels_by_sheet[sh] = labels
        preview_by_sheet[sh] = prev

    return labels_by_sheet, preview_by_sheet


# =============================================================================
# Candidate shortlisting (recall-heavy)
# =============================================================================
@dataclass
class ShortlistRule:
    include: List[str]
    exclude: List[str] = field(default_factory=list)
    preferred_sheet_types: List[str] = field(default_factory=list)  # "balance"/"cashflow"
    max_per_sheet: int = 60


SHORTLIST_RULES: Dict[str, ShortlistRule] = {
    "ACT": ShortlistRule(
        include=["total current assets", "current assets", "current asset", "curr assets"],
        exclude=["non-current", "noncurrent", "total assets"],
        preferred_sheet_types=["balance"],
    ),
    "CHE": ShortlistRule(
        include=[
            "cash and cash equivalents", "cash & cash equivalents", "cash and equivalents",
            "cash equivalents", "cash at bank", "cash",
        ],
        exclude=[
            "cash flow", "change in cash", "net cash",
            "trade receivable", "accounts receivable", "receivable",
        ],
        preferred_sheet_types=["balance"],
    ),
    "LCT": ShortlistRule(
        include=["total current liabilities", "current liabilities", "current liability", "curr liabilities"],
        exclude=["non-current", "noncurrent", "equity"],
        preferred_sheet_types=["balance"],
    ),
    "STD": ShortlistRule(
        include=[
            # classic debt in current liabilities
            "short-term debt", "short term debt",
            "short-term borrowings", "short term borrowings",
            "borrowings, current", "current borrowings",
            "interest-bearing debt, current", "interest bearing debt, current",
            "interest-bearing liabilities, current", "interest bearing liabilities, current",
            # current maturities phrasing
            "current maturities", "maturities - within 1 year", "within 1 year",
            "current portion of long-term debt", "current portion of long term debt",
            "current portion of debt", "current debt",
            "debt - long-term - maturities - within 1 year",
            # overdraft sometimes treated as borrowing
            "bank overdraft", "overdraft",
            # lease (included only if INCLUDE_LEASES_IN_STD=True)
            "lease liabilities, current", "current lease liabilities", "capital lease maturities",
        ],
        exclude=[
            "accounts payable", "trade payables", "other payables",
            "tax payable", "taxes payable",
            "accrued", "provision",
        ],
        preferred_sheet_types=["balance"],
    ),
    "TXP": ShortlistRule(
        include=[
            "tax payable", "taxes payable", "income tax payable", "income taxes payable",
            "current tax liability", "current tax liabilities",
            "current tax payable",
        ],
        exclude=["deferred", "tax expense", "income tax expense", "tax receivable", "tax asset", "provision"],
        preferred_sheet_types=["balance"],
    ),
    "PPEGT": ShortlistRule(
        include=[
            "property, plant and equipment", "property plant and equipment",
            "property, plant", "plant and equipment", "ppe",
            "tangible fixed assets", "tangible assets",
        ],
        exclude=["intangible", "goodwill", "right-of-use", "right of use", "investment property"],
        preferred_sheet_types=["balance"],
    ),
    "AT": ShortlistRule(
        include=["total assets", "assets - total", "assets, total"],
        exclude=["current assets", "non-current assets", "noncurrent assets"],
        preferred_sheet_types=["balance"],
    ),
    "OANCF": ShortlistRule(
        include=[
            "net cash flow from operating activities",
            "cash flow from operating activities",
            "net cash from operating activities",
            "net cash provided by operating activities",
            "net cash used in operating activities",
            "operating cash flow",
            "cash generated from operations",
        ],
        exclude=["investing", "financing", "free cash flow", "change in cash"],
        preferred_sheet_types=["cashflow"],
    ),
}


def _contains_any(text: str, needles: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(n) in t for n in needles)


def _contains_balancing(text: str) -> bool:
    t = _norm(text)
    return any(re.search(p, t) for p in FORBIDDEN_BALANCING_PATTERNS)


def shortlist_candidates(sheet_labels: Dict[str, List[str]], variable: str) -> List[Tuple[str, str]]:
    """
    Recall-heavy shortlist:
    - include-based substring matching
    - minimal excludes
    - prefer correct sheet types, but allow fallback
    """
    rule = SHORTLIST_RULES[variable]
    out: List[Tuple[str, str]] = []

    # pass 1: preferred sheet types
    for sheet, labels in sheet_labels.items():
        if rule.preferred_sheet_types and classify_sheet_type(sheet) not in rule.preferred_sheet_types:
            continue
        per_sheet = 0
        for lab in labels:
            if _contains_balancing(lab):
                if "balancing" in _norm(lab) or "remaining" in _norm(lab) or "subtotal" in _norm(lab):
                    continue
            if _contains_any(lab, rule.include) and not _contains_any(lab, rule.exclude):
                out.append((sheet, lab))
                per_sheet += 1
                if per_sheet >= rule.max_per_sheet:
                    break

    # pass 2: non-preferred sheets if too few
    if len(out) < 10:
        for sheet, labels in sheet_labels.items():
            if classify_sheet_type(sheet) in rule.preferred_sheet_types:
                continue
            for lab in labels:
                if _contains_any(lab, rule.include) and not _contains_any(lab, rule.exclude):
                    out.append((sheet, lab))
            if len(out) >= 80:
                break

    # pass 3: broad fallback (ignore excludes) if still too few
    if len(out) < 5:
        for sheet, labels in sheet_labels.items():
            for lab in labels:
                if _contains_any(lab, rule.include):
                    out.append((sheet, lab))
            if len(out) >= 80:
                break

    # de-dup preserve order
    seen = set()
    deduped = []
    for s, lab in out:
        key = (s, lab)
        if key not in seen:
            seen.add(key)
            deduped.append((s, lab))

    # Baseline: exclude leases from STD
    if variable == "STD" and not INCLUDE_LEASES_IN_STD:
        deduped = [(s, lab) for (s, lab) in deduped if "lease" not in _norm(lab)]

    return deduped[:80]


# =============================================================================
# JSON schema
# =============================================================================
MAPPING_SCHEMA = {
    "name": "accruals_row_mapping_v4",
    "schema": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "enum": ALL_VARS},
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


# =============================================================================
# Prompt builder
# =============================================================================
def build_prompt(
    sheet_preview: dict[str, list[list[str]]],
    shortlists: dict[str, list[tuple[str, str]]],
) -> str:
    lines: list[str] = []

    lines.append("You are mapping Balance Sheet and Cash Flow row labels to variables used for working-capital accruals.")
    lines.append("You MUST choose row labels EXACTLY as written in column A.")
    lines.append("You should choose from the provided candidate lists whenever possible.")
    lines.append("Return JSON ONLY that conforms to the given schema.")
    lines.append("IMPORTANT: Only Balance Sheet and Cash Flow sheets are provided (no income statement).")
    lines.append("")
    lines.append("Context: These are annual reports for Nordic listed firms, typically reported under IFRS.")
    lines.append("Under IFRS, balance sheet labels may use terms like 'Statement of financial position', 'Borrowings', and debt may be split into 'Borrowings, current' and 'Maturities within 1 year'.")
    lines.append("Use the label semantics and sheet type to choose the best matches.")
    lines.append("")

    lines.append("Manual review rule (IMPORTANT):")
    lines.append("- Set needs_manual_review=true ONLY when you are not confident in the final_choice (ambiguity/low confidence).")
    lines.append("- Do NOT set needs_manual_review=true merely because you chose a label that was not in the shortlist.")
    lines.append("- If you choose a label outside the shortlist but you are confident it is correct, keep needs_manual_review=false and explain briefly in notes.")
    lines.append("")

    lines.append("Targets:")
    for k, desc in ACCRUALS_TARGETS.items():
        lines.append(f"- {k}: {desc}")
    lines.append("")

    lines.append("General rules:")
    lines.append("1) Do NOT invent labels. Choose labels exactly as written.")
    lines.append("2) Avoid obvious balancing/subtotal lines when a cleaner reported line exists.")
    lines.append("3) Do NOT repeat the same row_label twice in final_choice.")
    lines.append("4) Labels can change over time: you may output multiple final_choice rows as a PRIORITY list (most preferred first).")
    lines.append("5) Use final_choice=[] only if the target truly does not exist or cannot be identified.")
    lines.append("")

    # STD special rule
    lines.append("STD rule (IMPORTANT):")
    lines.append("- STD corresponds to short-term debt/current maturities of long-term debt (financing-related current liabilities).")
    if INCLUDE_LEASES_IN_STD:
        lines.append("- Include current lease liabilities ONLY if they are clearly part of interest-bearing financing current liabilities (leases included version).")
    else:
        lines.append("- Baseline definition: EXCLUDE lease liabilities. Do NOT include rows containing 'lease' in STD.")
    lines.append("- Under IFRS, STD may be split across multiple rows (e.g., 'Borrowings, current' and 'Current maturities within 1 year').")
    lines.append("- Therefore, STD final_choice may contain MULTIPLE rows and will be treated as a SUM later.")
    lines.append("- Avoid double counting: do NOT select both a total current borrowings/interest-bearing debt line AND its component maturities.")
    lines.append("- Prefer: (a) if a clean total exists ('Borrowings, current' / 'Interest-bearing debt, current'), select ONLY that.")
    lines.append("- Otherwise select the minimal set of components that together represent STD (e.g., current maturities + short-term borrowings).")
    lines.append("- Do NOT include trade/other payables, accrued liabilities, or taxes payable in STD.")
    lines.append("")

    lines.append("Missing data rule (IMPORTANT):")
    lines.append("- If a target variable truly does not exist in the statements, set final_choice=[].")
    lines.append("- For these balance sheet / cash flow variables, missing is NOT automatically interpreted as 0.")
    lines.append("- Set needs_manual_review=true ONLY when you suspect the item exists but you cannot identify it confidently.")
    lines.append("")

    lines.append("IMPORTANT DATA RULE:")
    lines.append("- A row label can be valid even if the A–F preview shows blanks in recent columns.")
    lines.append("- Do NOT reject a semantically correct row only because recent preview values are blank (labels may change over time).")
    lines.append("")

    lines.append("Context: Below is a preview of the first 6 columns (A–F) from each sheet.")
    lines.append("Column A is the row label. Columns B–F are recent values.")
    lines.append("")

    for sheet, rows in sheet_preview.items():
        lines.append(f"\n=== SHEET: {sheet} (A–F preview) ===")
        for r in rows[:200]:
            lines.append(" | ".join(r))
        if len(rows) > 200:
            lines.append(f"... ({len(rows) - 200} more rows omitted)")
    lines.append("")

    lines.append("Candidate lists (use these to help you find the best labels; you may choose outside them if needed):")
    for var, cands in shortlists.items():
        lines.append(f"\n--- CANDIDATES FOR {var} ---")
        if not cands:
            lines.append("(no candidates found)")
            continue
        for sh, lab in cands:
            lines.append(f"- [{sh}] {lab}")

    return "\n".join(lines)


# =============================================================================
# OpenAI call helper (structured outputs)
# =============================================================================
def _responses_create_json_schema(model: str, prompt: str, reasoning_effort: str) -> dict:
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


# =============================================================================
# Post-processing: validate + confidence-based manual review
# =============================================================================
def validate_mapping_against_labels(mapping: dict, sheet_labels: dict[str, list[str]]) -> dict:
    """
    Ensure every (sheet_name, row_label) in final_choice exists in the workbook labels.
    Invalid choices are removed and the variable is flagged for manual review.
    """
    labels_set = {s: set(labs) for s, labs in sheet_labels.items()}

    for v in mapping.get("variables", []):
        cleaned = []
        for ch in v.get("final_choice", []):
            s = ch.get("sheet_name", "")
            lab = ch.get("row_label", "")
            if s in labels_set and lab in labels_set[s]:
                cleaned.append(ch)
            else:
                v["needs_manual_review"] = True
                msg = f"Chosen label not found in sheet: [{s}] {lab}"
                v["notes"] = (v.get("notes", "") + " | " + msg).strip(" |")
        v["final_choice"] = cleaned

    return mapping


def enforce_confidence_manual_review(mapping: dict, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Set needs_manual_review based on confidence/uncertainty only (NOT on shortlist membership).

    - If the model already set needs_manual_review=True, keep it.
    - If final_choice is empty, do not auto-flag (can be truly missing).
    - Otherwise, if best candidate confidence < threshold, flag.
    """
    for v in mapping.get("variables", []):
        if v.get("needs_manual_review", False):
            continue

        finals = v.get("final_choice", [])
        if not finals:
            continue

        cands = v.get("candidates", [])
        if not cands:
            continue

        best_conf = 0.0
        for c in cands:
            try:
                best_conf = max(best_conf, float(c.get("confidence", 0.0)))
            except Exception:
                pass

        if best_conf < confidence_threshold:
            v["needs_manual_review"] = True
            msg = f"Auto-flag: best candidate confidence {best_conf:.2f} < {confidence_threshold:.2f}"
            v["notes"] = (v.get("notes", "") + " | " + msg).strip(" |")

    return mapping


# =============================================================================
# Public API
# =============================================================================
def llm_map_accruals_rows(
    xlsx_path: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict:
    """
    Maps Balance Sheet + Cash Flow rows to accrual variables.
    STD final_choice is intended to be SUMMED later.
    Other variables are intended as PRIORITY lists later.
    Manual review is based on confidence/uncertainty (not shortlist membership).
    """
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    sheet_labels, sheet_preview = read_columns_a_to_f_balance_and_cashflow(xlsx_path)

    shortlists: Dict[str, List[Tuple[str, str]]] = {}
    for var in ALL_VARS:
        shortlists[var] = shortlist_candidates(sheet_labels, var)

    prompt = build_prompt(sheet_preview, shortlists)
    mapping = _responses_create_json_schema(model=model, prompt=prompt, reasoning_effort=reasoning_effort)

    mapping = validate_mapping_against_labels(mapping, sheet_labels)
    mapping = enforce_confidence_manual_review(mapping, confidence_threshold=CONFIDENCE_THRESHOLD)

    return mapping


def save_mapping_accruals(mapping: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")