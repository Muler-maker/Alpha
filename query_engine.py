import os
import json
import re
import copy
import ast
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st      # ← ADD THIS
from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate

def _norm_key(series: pd.Series) -> pd.Series:
    """
    Normalize join keys to prevent mismatches due to casing/spacing.
    """
    return (
        series.astype(str)
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )
import datetime as _dt

def _iso_week_to_date(year: int, week: int, weekday: int = 1) -> _dt.date:
    """
    Return a date for ISO year/week/weekday.
    weekday: 1=Mon .. 7=Sun
    """
    # Python's fromisocalendar uses ISO week rules (recommended)
    return _dt.date.fromisocalendar(int(year), int(week), int(weekday))

def _ensure_period_columns(df: pd.DataFrame, year_col: str, week_col: str) -> pd.DataFrame:
    """
    Adds:
      - _WeekDate (Monday of ISO week)
      - Month (1-12)
      - Quarter (1-4)
      - YearMonth (e.g., 2025-02)
      - YearQuarter (e.g., 2025-Q1)
    """
    out = df.copy()

    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out[week_col] = pd.to_numeric(out[week_col], errors="coerce")
    out = out.dropna(subset=[year_col, week_col])
    out[year_col] = out[year_col].astype(int)
    out[week_col] = out[week_col].astype(int)

    # ISO week Monday date
    out["_WeekDate"] = [
        _iso_week_to_date(y, w, 1) for y, w in zip(out[year_col].tolist(), out[week_col].tolist())
    ]

    out["Month"] = [d.month for d in out["_WeekDate"]]
    out["Quarter"] = [((d.month - 1) // 3) + 1 for d in out["_WeekDate"]]

    out["YearMonth"] = [f"{d.year:04d}-{d.month:02d}" for d in out["_WeekDate"]]
    out["YearQuarter"] = [f"{d.year:04d}-Q{((d.month - 1)//3)+1}" for d in out["_WeekDate"]]

    return out

# --- Environment & OpenAI client ---

load_dotenv()

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_API_KEY")   # ← read from Streamlit secrets too
)

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None



# --- Business rules you defined ---

COUNTABLE_STATUSES = [
    "Shipped",
    "Partially shipped",
    "Shipped and arrived late",
    "Order being processed",
]

CANCELLED_STATUSES = [
    "Rejected \\ Cancelled by Isotopia",
    "Cancelled by the customer",
    "Rejected \\ Cancelled",
]

# Map conceptual fields (customer, year, etc.) → actual column names in the DF.
FIELD_CANDIDATES = {
    # End customer (hospital/clinic) – Company name / The customer
    "customer": ["The customer", "Customer", "Company name"],

    # Unique order quantity (actuals)
    "total_mci": ["Total_mCi"],

    # Projection quantity (from Projections sheet, joined in consolidated.py)
    "proj_mci": ["Proj_Amount"],

    # Shipping status
    "shipping_status": ["Shipping Status", "ShippingStatus"],

    # Time / geography
    "year": ["Year"],
    "month": ["Month"],
    "quarter": ["Quarter"],
    "half_year": ["Half year"],
    "country": ["Country"],
    "region": ["Region (from Company name)", "Region"],
        # Primary week used for analysis = Week number for Activity vs Projection
    # Canonical week for all analysis = 'Week'
    # (renamed from "Week number for Activity vs Projection" in preprocess_orders)
    "week": [
        "Week",                              # ✅ canonical
        "Week number for Activity vs Projection",
        "Week of supply",
        "Week of Supply",
        "Week_of_supply",
        "Updated week number",
    ],


    # Distributor entity
    "distributor": [
        "Distributing company (from Company name)",
        "Distributor",
    ],

    # PRODUCTS:
    # What we SOLD as (billing view)
    "product_sold": ["Catalogue description (sold as)", "Product"],

    # What we PRODUCED (technical / production)
    "product_catalogue": ["Catalogue description"],

    # Production site
    "production_site": ["Production site"],
}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name that exists in df from the candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Build a mapping {conceptual_field -> actual_dataframe_column_name_or_None}."""
    return {key: _find_col(df, cands) for key, cands in FIELD_CANDIDATES.items()}


# --------------------------------------------------------------------
# 1) INTERPRETATION LAYER – turn NL question → JSON spec
# -----------------------
def _expand_distributor_name(text: str) -> Optional[str]:
    """
    Map short names to full distributor names.
    Returns a SEARCH PATTERN, not an exact name.
    """
    text = text.strip().lower()
    
    # Map input patterns to search keywords (what we'll match in the data)
    mapping = {
        "dsd": "dsd",
        "dsd pharma": "dsd",
        "pi medical": "pi medical",
        "pi": "pi medical",
        "scantor": "scantor",
        "scintomics": "scintomics",
        "sinotau": "sinotau",
    }
    
    if text in mapping:
        return mapping[text]
    
    for key, value in mapping.items():
        if key in text or text in key:
            return value
    
    # If no match, return the text as-is (case-insensitive search)
    return text


def _detect_entity_comparison_from_text(question: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process question text to detect comparison patterns.
    Handles: "Compare X to Y", "X vs Y", etc.
    """
    q_lower = (question or "").lower()
    
    # Skip if already marked as comparison
    if spec.get("aggregation") == "compare":
        return spec
    
    # Pattern 1: "Compare X to Y"
    compare_pattern = r"compare\s+([a-z\s]+?)\s+(?:to|vs|versus)\s+([a-z\s]+?)(?:\s+in\s+|$)"
    m = re.search(compare_pattern, q_lower)
    
    if m:
        entity_a = m.group(1).strip()
        entity_b = m.group(2).strip()
        print(f"[DETECT] Matched 'Compare X to Y': '{entity_a}' vs '{entity_b}'")
        
        entity_a = _expand_distributor_name(entity_a)
        entity_b = _expand_distributor_name(entity_b)
        
        if entity_a and entity_b:
            spec["aggregation"] = "compare"
            compare = spec.get("compare") or {}
            compare["entities"] = [entity_a, entity_b]
            compare["entity_type"] = "distributor"
            spec["compare"] = compare
            
            filters = spec.get("filters") or {}
            filters["distributor"] = None
            spec["filters"] = filters
            
            spec["_question_text"] = question
            return spec
    
    # Pattern 2: "X vs Y"
    vs_pattern = r"\b([a-z\s]+?)\s+(?:vs|versus)\s+([a-z\s]+?)(?:\s+in\s+|$)"
    m = re.search(vs_pattern, q_lower)
    
    if m:
        entity_a = m.group(1).strip()
        entity_b = m.group(2).strip()
        
        entity_a_expanded = _expand_distributor_name(entity_a)
        entity_b_expanded = _expand_distributor_name(entity_b)
        
        if entity_a_expanded and entity_b_expanded:
            print(f"[DETECT] Matched 'X vs Y': '{entity_a_expanded}' vs '{entity_b_expanded}'")
            
            spec["aggregation"] = "compare"
            compare = spec.get("compare") or {}
            compare["entities"] = [entity_a_expanded, entity_b_expanded]
            compare["entity_type"] = "distributor"
            spec["compare"] = compare
            
            filters = spec.get("filters") or {}
            filters["distributor"] = None
            spec["filters"] = filters
            
            spec["_question_text"] = question
            return spec
    
    return spec

def _interpret_question_with_llm(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Use GPT to translate the user's natural language question into
    a structured query spec.
    """
    if client is None:
        return _interpret_question_fallback(question, history=history)

    system_prompt = """
You are an assistant called *Isotopia Alpha* that translates natural-language questions
about radiopharmaceutical orders into a structured JSON query specification.

You DO NOT see the actual data. You only decide WHAT to filter and aggregate.

Return ONLY a single valid JSON object with no markdown. Use this exact schema:

{
  "aggregation": "sum_mci",
  "group_by": [],
  "filters": {
    "year": null,
    "week": null,
    "month": null,
    "quarter": null,
    "half_year": null,
    "customer": null,
    "distributor": null,
    "country": null,
    "region": null,
    "product_sold": null,
    "product_catalogue": null,
    "production_site": null
  },
  "shipping_status_mode": "countable",
  "shipping_status_list": [],
  "time_window": {
    "mode": null,
    "n_weeks": null,
    "anchor": {
      "year": null,
      "week": null
    }
  },
  "compare": {
    "period_a": null,
    "period_b": null
  },
  "explanation": ""
}

ALLOWED aggregation values:
- "sum_mci"       → sum of Total_mCi (default)
- "average_mci"   → average Total_mCi per row
- "share_of_total"→ share of total volume under the same time/product filters
- "growth_rate"   → growth from one period to another

GROWTH RATE DETECTION - CRITICAL

If the user asks for:
- "growth", "growth rate", "growth rates"
- "weekly growth", "week-over-week", "WoW"
- "yearly growth", "year-over-year", "YoY"
- "how much did X grow"
- "growth per week"
- "growth by week"
- "from 2024 to 2025" or "compared to 2024"

THEN: aggregation = "growth_rate"

For weekly growth: group_by should include "week"
For yearly growth: group_by should include "year"

IMPORTANT: "Compare the weekly growth" means growth_rate aggregation with weekly breakdown.
Do NOT use aggregation = "compare" for growth contexts.

ENTITIES
- "customer": end-customer (hospital/clinic)
- "distributor": distributing company
- "country": customer's country
- "region": Region (from Company name)

TIME FILTERS
- "year": integer 2023, 2024, 2025
- "week": integer 1-52
- "month": integer 1-12
- "quarter": "Q1", "Q2", "Q3", "Q4"
- "half_year": "H1", "H2"

PRODUCT DIMENSIONS
1) product_sold: what was ordered/billed
2) product_catalogue: what was manufactured

SHIPPING STATUS LOGIC
- "countable": shipped, partially shipped, shipped late, being processed (default)
- "cancelled": rejected or cancelled
- "all": all statuses
- "explicit": specific statuses listed by user

GROUP BY LOGIC
Set group_by for breakdowns:
- "per year" → ["year"]
- "per country" → ["country"]
- "per distributor" → ["distributor"]
- "per customer" → ["customer"]
- "per week" or "by week" or "weekly" → ["week"]
- "weekly breakdown" → ["week"] or ["year", "week"]
- Multiple dimensions allowed: ["customer", "year"], ["distributor", "product_sold"]

ALWAYS:
- If unsure, leave a filter as null.
- Always set "shipping_status_mode".
- Return JSON ONLY with no markdown.
"""

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        for m in history[-6:]:
            role = m.get("role")
            if role in ("user", "assistant"):
                content = m.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content
        spec = json.loads(raw)
    except Exception as e:
        st.warning(f"LLM error: {e}")
        spec = _interpret_question_fallback(question)

    # ===== CRITICAL: POST-PROCESS FOR DISTRIBUTOR COMPARISON =====
    spec = _detect_entity_comparison_from_text(question, spec)

    # SAFETY: Ensure spec has all required keys with proper types
    if spec is None:
        spec = {}
    
    spec.setdefault("filters", {})
    spec.setdefault("group_by", [])
    spec.setdefault("time_window", {})
    spec.setdefault("compare", {})
    spec.setdefault("shipping_status_mode", "countable")
    spec.setdefault("shipping_status_list", [])
    
    # Ensure sub-dicts are dicts, not None
    if spec["filters"] is None:
        spec["filters"] = {}
    if spec["time_window"] is None:
        spec["time_window"] = {}
    if spec["compare"] is None:
        spec["compare"] = {}
    
# Apply any existing date heuristics
    spec = _augment_spec_with_date_heuristics(question, spec)

    # ------------------------------------------------------------------
    # PROJECTION VS ACTUAL DETECTION (post-process)
    # ------------------------------------------------------------------
    q_lower = str(question or "").lower()

    # Check for projection vs actual patterns
    has_projection = any(kw in q_lower for kw in [
        "projection", "projections", "projected", "forecast", "forecasted",
        "budget", "budgeted", "plan", "planned", "estimate"
    ])

    has_actual = any(kw in q_lower for kw in [
        "actual", "actuals", "realized", "real", "executed", "delivered"
    ])

    has_comparison = any(kw in q_lower for kw in [
        "vs", "versus", "compared to", "compare", "vs.", "v.",
        "against", "difference", "variance", "gap", "delta"
    ])

    # If we have projection + actual + comparison, force this aggregation
    if has_projection and has_actual and has_comparison:
        print(f"🔴 FORCING projection_vs_actual aggregation (detected keywords)")
        spec["aggregation"] = "projection_vs_actual"
        
        # Ensure we group by week if mentioned
        if "week" in q_lower and "week" not in (spec.get("group_by") or []):
            gb = spec.get("group_by") or []
            gb.append("week")
            spec["group_by"] = gb
            print(f"  Added 'week' to group_by")
        
        spec["_question_text"] = question
        return spec

    # Extra post-processing for patterns that the LLM often misses
    q_lower = str(question or "").lower()  # SAFETY: ensure q_lower is string
    
    # Always keep the raw question text for downstream helpers
    spec["_question_text"] = question

    # Extra post-processing for patterns that the LLM often misses
    q_lower = str(question or "").lower()  # SAFETY: ensure q_lower is string
    
    # Always keep the raw question text for downstream helpers
    spec["_question_text"] = question
    
    # ===== CRITICAL: OVERRIDE FOR COMPARISON DETECTION =====
    # Check for comparison patterns BEFORE other logic
    has_compare_word = any(word in q_lower for word in ["vs", "versus", "compare", "compared to"])
    has_distributor_names = any(name in q_lower for name in ["dsd", "pi medical", "scantor", "scintomics"])
    
    if has_compare_word and has_distributor_names:
        # Extract distributor names
        entities_found = []
        if "pi medical" in q_lower:
            entities_found.append("PI Medical")
        if "dsd" in q_lower:
            entities_found.append("DSD")
        if "scantor" in q_lower:
            entities_found.append("Scantor")
        if "scintomics" in q_lower:
            entities_found.append("Scintomics")
        
        if len(entities_found) >= 2:
            print(f"🔴 OVERRIDE: Detected distributor comparison: {entities_found}")
            spec["aggregation"] = "compare"
            
            compare = spec.get("compare") or {}
            if not isinstance(compare, dict):
                compare = {}
            
            compare["entities"] = entities_found
            compare["entity_type"] = "distributor"
            spec["compare"] = compare
            
            # Remove any distributor filter so we get both entities
            filters = spec.get("filters") or {}
            filters["distributor"] = None
            spec["filters"] = filters
            
            spec["_question_text"] = question
            return spec
    
    # Projection vs actual questions
    proj_keywords = ["projection", "projections", "forecast", "budget", "plan"]
    actual_keywords = ["actual", "actuals", "vs", "versus", "variance", "gap"]

    if any(k in q_lower for k in proj_keywords) and any(k in q_lower for k in actual_keywords):
        spec["aggregation"] = "projection_vs_actual"

    # GROWTH RATE DETECTION & COMPARISON PATTERNS
    growth_keywords = [
        "growth rate", "growth rates", "weekly growth", "week-over-week",
        "wow", "yoy", "year-over-year", "yearly growth", "growth per week",
        "growth by week", "last", "previous"
    ]
    
    if any(kw in q_lower for kw in growth_keywords):
        print(f"🔴 FORCING growth_rate aggregation (detected keywords)")
        spec["aggregation"] = "growth_rate"
        
        gb = spec.get("group_by") or []
        if not isinstance(gb, list):
            gb = []
        
        # Check for "last N weeks vs previous N weeks" pattern (2-period comparison)
        m_compare = re.search(
            r"(last|previous)\s+(\d+)\s+weeks?\s+(?:vs|versus|compared to|to)\s+(?:the\s+)?(previous|last)\s+(\d+)\s+weeks?",
            q_lower
        )
        if m_compare:
            print(f"🔴 DETECTED: Last/Previous N weeks vs Previous M weeks comparison")
            n_weeks_a = int(m_compare.group(2))
            n_weeks_b = int(m_compare.group(4))
            
            tw = spec.get("time_window") or {}
            if not isinstance(tw, dict):
                tw = {}
            
            tw["mode"] = "compare_periods"
            tw["period_a"] = {
                "mode": "last_n_weeks",
                "n_weeks": n_weeks_a,
                "anchor": {"year": None, "week": None}
            }
            tw["period_b"] = {
                "mode": "last_n_weeks",
                "n_weeks": n_weeks_b + n_weeks_a,
                "anchor": {"year": None, "week": None}
            }
            spec["time_window"] = tw
            
            if "week" not in gb:
                gb.insert(0, "week")
            print(f"🔴 Set up 2-period comparison: {n_weeks_a} weeks vs {n_weeks_b} weeks prior")
        else:
            # Single period: "last N weeks" / "previous N weeks"
            m_window = re.search(r"(last|previous)\s+(\d+)\s+weeks?", q_lower)
            if m_window:
                print(f"🔴 DETECTED: Last/previous N weeks pattern")
                n_weeks = int(m_window.group(2))
                tw = spec.get("time_window") or {}
                if not isinstance(tw, dict):
                    tw = {}
                tw["mode"] = "last_n_weeks"
                tw["n_weeks"] = n_weeks
                tw["anchor"] = {"year": None, "week": None}
                spec["time_window"] = tw
                
                if "week" not in gb:
                    gb.insert(0, "week")
                print(f"🔴 Set time_window mode='last_n_weeks', n_weeks={n_weeks}")
        
        spec["group_by"] = gb
        spec["_question_text"] = question
        return spec

    # TOP N DETECTION
    top_n_keywords = [
        "top ", "top10", "top 10", "top5", "top 5", "top3", "top 3",
        "highest", "largest", "biggest", "most ordered", "most volume",
    ]
    
    top_n_detected = any(kw in q_lower for kw in top_n_keywords)
    
    if top_n_detected:
        print(f"🔴 DETECTED: Top N question")
        spec["aggregation"] = "top_n"
        spec["_question_text"] = question
        return spec

    # Default DSD heuristic fix (only if no comparison detected)
    filters = spec.get("filters") or {}
    if "dsd" in q_lower and "pi medical" not in q_lower:
        if filters.get("customer"):
            filters["customer"] = None
        if not filters.get("distributor"):
            filters["distributor"] = "DSD"
    spec["filters"] = filters
    
    spec["_question_text"] = question
    return spec

def _interpret_question_fallback(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Very simple heuristic fallback when OpenAI is not available."""
    spec: Dict[str, Any] = {
        "aggregation": "sum_mci",
        "group_by": [],
        "filters": {
            "year": None,
            "week": None,
            "month": None,
            "quarter": None,
            "half_year": None,
            "customer": None,
            "distributor": None,
            "country": None,
            "region": None,
            "product_sold": None,
            "product_catalogue": None,
            "production_site": None,
        },
        "shipping_status_mode": "countable",
        "shipping_status_list": [],
        "compare": {
            "period_a": None,
            "period_b": None,
            "entities": None,
            "entity_type": None,
            "mode": None,
            "table_group_by": None,
        },
        "time_window": {
            "mode": None,
            "n_weeks": None,
            "anchor": {"year": None, "week": None},
        },
        "explanation": "Heuristic interpretation without LLM.",
    }

    q_lower = (question or "").lower()

    # Check for growth rate keywords
    growth_keywords = ["growth rate", "growth rates", "weekly growth", "week-over-week", "wow", "yoy", "year-over-year"]
    if any(kw in q_lower for kw in growth_keywords):
        spec["aggregation"] = "growth_rate"
        if "weekly" in q_lower or "week" in q_lower:
            spec["group_by"].append("week")
        elif "yearly" in q_lower or "year" in q_lower:
            spec["group_by"].append("year")
    elif (
        ("projection" in q_lower or "projections" in q_lower
         or "forecast" in q_lower or "budget" in q_lower)
        and ("actual" in q_lower or "vs" in q_lower
             or "versus" in q_lower or "variance" in q_lower)
    ):
        spec["aggregation"] = "projection_vs_actual"
    elif "compare" in q_lower:
        spec["aggregation"] = "compare"
    else:
        spec["aggregation"] = "sum_mci"

    spec["_question_text"] = question or ""

    # Year detection
    m = re.search(r"(20[2-3][0-9])", q_lower)
    if m:
        spec["filters"]["year"] = int(m.group(1))

    # Cancellation detection
    if "cancel" in q_lower or "reject" in q_lower:
        spec["shipping_status_mode"] = "cancelled"

    return spec
   
def _augment_spec_with_date_heuristics(question: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process the spec from the LLM with simple regex/keyword rules
    to ensure year / month / quarter / half_year are filled when obvious
    in the question text, and to enforce dynamic week windows when the
    LLM forgets to set time_window (e.g. 'last 4 weeks').
    """
    if spec is None:
        spec = {}
    
    q = str(question or "").lower()  # SAFETY: ensure q is string
    
    # Ensure all nested dicts exist and are dicts
    filters = spec.get("filters")
    if filters is None or not isinstance(filters, dict):
        filters = {}
    spec["filters"] = filters

    time_window = spec.get("time_window")
    if time_window is None or not isinstance(time_window, dict):
        time_window = {}
    spec["time_window"] = time_window
    
    # Ensure anchor exists
    if "anchor" not in time_window:
        time_window["anchor"] = {"year": None, "week": None}

    # -----------------------
    # YEAR
    # -----------------------
    if not filters.get("year"):
        m = re.search(r"\b(20[2-3][0-9])\b", q)
        if m:
            try:
                filters["year"] = int(m.group(1))
            except ValueError:
                pass

    # -----------------------
    # MONTH
    # -----------------------
    if not filters.get("month"):
        month_map = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        for name, num in month_map.items():
            if name in q:
                filters["month"] = num
                break

    # -----------------------
    # QUARTER
    # -----------------------
    if not filters.get("quarter"):
        m = re.search(r"\bq([1-4])\b", q)
        if m:
            filters["quarter"] = f"Q{m.group(1)}"
        else:
            quarter_words = {
                "first quarter": "Q1",
                "1st quarter": "Q1",
                "second quarter": "Q2",
                "2nd quarter": "Q2",
                "third quarter": "Q3",
                "3rd quarter": "Q3",
                "fourth quarter": "Q4",
                "4th quarter": "Q4",
            }
            for phrase, qval in quarter_words.items():
                if phrase in q:
                    filters["quarter"] = qval
                    break

    # -----------------------
    # HALF-YEAR
    # -----------------------
    if not filters.get("half_year"):
        m = re.search(r"\bh([12])\b", q)
        if m:
            filters["half_year"] = f"H{m.group(1)}"
        else:
            if "first half" in q or "1st half" in q:
                filters["half_year"] = "H1"
            elif "second half" in q or "2nd half" in q:
                filters["half_year"] = "H2"

    # -----------------------
    # TIME WINDOW (dynamic weeks)
    # -----------------------
    # Only add our heuristics if LLM did NOT already set a mode
    if not time_window.get("mode"):
        # --- last / past / previous / trailing / rolling N weeks ---
        m_last_weeks = re.search(
            r"(last|past|previous|trailing|rolling)\s+(\d+)\s+weeks?",
            q,
        )
        if m_last_weeks:
            n_weeks_str = m_last_weeks.group(2)
            try:
                n_weeks = int(n_weeks_str)
                time_window["mode"] = "last_n_weeks"
                time_window["n_weeks"] = n_weeks
                time_window["anchor"]["year"] = None
                time_window["anchor"]["week"] = None
                # IMPORTANT: dynamic window → we must NOT also use filters.week
                filters["week"] = None
            except (ValueError, TypeError):
                pass

    # Anchored window: "N weeks before week X [of YEAR]"
    if not time_window.get("mode"):
        m_anchor = re.search(
            r"(\d+)\s+weeks?\s+(?:before|leading up to)\s+week\s+(\d+)(?:\s+of\s+(20[2-3][0-9]))?",
            q,
        )
        if m_anchor:
            try:
                n_weeks = int(m_anchor.group(1))
                week_x = int(m_anchor.group(2))
            except (ValueError, TypeError):
                n_weeks = None
                week_x = None

            if n_weeks is not None and week_x is not None:
                time_window["mode"] = "anchored_last_n_weeks"
                time_window["n_weeks"] = n_weeks
                time_window["anchor"]["week"] = week_x

                year_str = m_anchor.group(3)
                if year_str:
                    try:
                        year_val = int(year_str)
                        time_window["anchor"]["year"] = year_val
                        filters["year"] = year_val
                    except (ValueError, TypeError):
                        pass

                # For anchored windows we also clear filters.week
                filters["week"] = None

    spec["filters"] = filters
    spec["time_window"] = time_window

    # --- ENSURE WEEK EXTRACTION ALWAYS HAPPENS ---
    if not filters.get("week"):
        m = re.search(r"week\s+(\d{1,2})", q)
        if m:
            try:
                filters["week"] = int(m.group(1))
            except (ValueError, TypeError):
                pass

    # --- ENSURE YEAR EXTRACTION ALWAYS HAPPENS ---
    if not filters.get("year"):
        m = re.search(r"\b(20[2-3][0-9])\b", q)
        if m:
            filters["year"] = int(m.group(1))

    # --- FINAL: if BOTH year & week exist, enable metadata ---
    spec["_metadata_ready"] = (
        filters.get("year") is not None and
        filters.get("week") is not None
    )

    return spec
def _normalize_entity_filters(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a customer value actually exists in the distributor column,
    move it to distributor. This fixes cases where the LLM or injection
    logic misclassified a distributor as a customer.
    
    Also checks the reverse: if a distributor value is actually a customer.
    """
    filters = spec.get("filters") or {}
    
    # Build set of distributor names (normalized)
    if "Distributor" in df.columns:
        dist_names = {
            str(x).strip().lower() 
            for x in df["Distributor"].dropna().unique()
        }
    else:
        dist_names = set()
    
    # Build set of customer names (normalized)
    if "Customer" in df.columns:
        cust_names = {
            str(x).strip().lower() 
            for x in df["Customer"].dropna().unique()
        }
    else:
        cust_names = set()
    
    # Check if customer is actually a distributor
    cust_val = filters.get("customer")
    if cust_val:
        cust_lower = str(cust_val).strip().lower()
        if cust_lower in dist_names:
            print(f"🔧 NORMALIZING: Moving '{cust_val}' from customer → distributor (found in Distributor column)")
            filters["distributor"] = cust_val
            filters["customer"] = None
    
    # Check if distributor is actually a customer
    dist_val = filters.get("distributor")
    if dist_val:
        dist_lower = str(dist_val).strip().lower()
        if dist_lower in cust_names and dist_lower not in dist_names:
            print(f"🔧 NORMALIZING: Moving '{dist_val}' from distributor → customer (found in Customer column only)")
            filters["customer"] = dist_val
            filters["distributor"] = None
    
    spec["filters"] = filters
    return spec

def _disambiguate_customer_vs_distributor(
    df: pd.DataFrame, spec: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make a best-effort distinction between 'customer' and 'distributor'.

    Rules:
      - If question mentions 'distributor', prefer distributor filter.
      - If question mentions 'customer' (and not 'distributor'), prefer customer filter.
      - If a 'customer' value matches or nearly matches a known distributor name,
        treat it as a distributor (especially important because projections live
        at distributor level).
    """
    filters = spec.get("filters") or {}
    question = (spec.get("_question_text") or "").lower()

    customer_val = filters.get("customer")
    distributor_val = filters.get("distributor")

    # 1) Explicit language in question
    mentions_distributor = "distributor" in question or "distributors" in question
    mentions_customer = "customer" in question or "customers" in question

    if mentions_distributor and not mentions_customer:
        # If we already have a distributor filter, drop any accidental customer
        if distributor_val:
            filters.pop("customer", None)
            spec["filters"] = filters
            return spec
        # If we DON'T have distributor_val yet, fall through so we can
        # still promote customer -> distributor later.

    if mentions_customer and not mentions_distributor:
        # If we already have a customer filter, drop any accidental distributor
        if customer_val:
            filters.pop("distributor", None)
            spec["filters"] = filters
            return spec
        # If we DON'T have customer_val yet, fall through.

    # 2) Build distributor & customer name sets for normalization
    dist_values: List[str] = []
    if "Distributor" in df.columns:
        dist_values = [str(x).strip() for x in df["Distributor"].dropna().unique()]
        distributor_names = {d.lower() for d in dist_values}
    else:
        distributor_names = set()

    cust_values: List[str] = []
    if "Customer" in df.columns:
        cust_values = [str(x).strip() for x in df["Customer"].dropna().unique()]
        customer_names = {c.lower() for c in cust_values}
    else:
        customer_names = set()

    # Helper for fuzzy token-based matching against distributors
    def _best_distributor_match(name: str) -> Optional[str]:
        name_norm = str(name).strip().lower()
        name_tokens = {t for t in name_norm.replace("-", " ").split() if len(t) > 1}
        if not name_tokens or not dist_values:
            return None

        best_match = None
        best_score = 0

        for dist in dist_values:
            dist_tokens = {t for t in dist.lower().replace("-", " ").split() if len(t) > 1}
            overlap = name_tokens & dist_tokens
            score = len(overlap)
            if score > best_score:
                best_score = score
                best_match = dist

        return best_match if best_score > 0 else None

    # 3) Customer → Distributor promotion (this is what fixes PI Medical)
    if customer_val and not distributor_val:
        cust_norm = str(customer_val).strip().lower()

        # 3a) Exact match against distributor names
        if cust_norm in distributor_names:
            canonical = next(
                (d for d in dist_values if d.lower() == cust_norm),
                customer_val,
            )
            print(
                f"🔧 NORMALIZING: Moving '{customer_val}' from customer → distributor "
                f"(exact match: '{canonical}')"
            )
            filters.pop("customer", None)
            filters["distributor"] = canonical
            spec["filters"] = filters
            return spec

        # 3b) Fuzzy match against distributor names (token overlap)
        best_match = _best_distributor_match(customer_val)
        if best_match:
            print(
                f"🔧 NORMALIZING: Treating customer '{customer_val}' as distributor '{best_match}' "
                f"(token overlap)"
            )
            filters.pop("customer", None)
            filters["distributor"] = best_match
            spec["filters"] = filters
            return spec

    # 4) Distributor → Customer fallback (rare, but keep the original safety)
    if distributor_val:
        dist_norm = str(distributor_val).strip().lower()
        if dist_norm in customer_names and dist_norm not in distributor_names:
            print(
                f"🔧 NORMALIZING: Moving '{distributor_val}' from distributor → customer "
                "(found only in Customer column)"
            )
            filters["customer"] = distributor_val
            filters["distributor"] = None

    spec["filters"] = filters
    return spec


def _inject_customer_from_question(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to infer entity filters from the question text.
    PRIORITY ORDER:
    1. If distributor filter is already set, skip customer injection
    2. Try to match as a DISTRIBUTOR first (check Distributor column) - AGGRESSIVE matching
    3. Fall back to matching as a CUSTOMER (check Customer column)
    """
    filters = spec.get("filters") or {}
    question = (spec.get("_question_text") or "").lower()
    
    # 🔧 DEBUG: Print all distributors and customers
    print(f"\n🔧 DEBUG _inject_customer_from_question():")
    print(f"  Question: {question}")
    print(f"  Current filters: {filters}")
    
    if "Distributor" in df.columns:
        unique_distributors = [str(x).strip() for x in df["Distributor"].dropna().unique()]
        print(f"  DISTRIBUTORS in data: {unique_distributors}")
    else:
        print(f"  ❌ No 'Distributor' column in df")
    
    if "Customer" in df.columns:
        unique_customers = [str(x).strip() for x in df["Customer"].dropna().unique()]
        print(f"  CUSTOMERS in data (first 10): {unique_customers[:10]}")
    else:
        print(f"  ❌ No 'Customer' column in df")
    
    # If distributor is already set, don't try to find a customer
    if filters.get("distributor"):
        print(f"  ✅ Distributor already set: {filters['distributor']} → skipping entity injection")
        return spec
    
    # If customer is already set, don't override it
    if filters.get("customer"):
        print(f"  ✅ Customer already set: {filters['customer']}")
        return spec

    print(f"  Trying to extract entity (distributor first, then customer)...")

    # ===== STEP 1: Try to match as DISTRIBUTOR (VERY AGGRESSIVE) =====
    if "Distributor" in df.columns:
        unique_distributors = [str(x).strip() for x in df["Distributor"].dropna().unique()]
        print(f"  Step 1: Checking {len(unique_distributors)} unique distributors...")
        
        # Tokenize question into keywords
        q_tokens = set(
            t.strip().lower() 
            for t in question.replace(",", " ").replace("'", "").split() 
            if len(t.strip()) > 2
        )
        print(f"    Question tokens: {q_tokens}")
        
        # Try exact match first (case-insensitive)
        for dist in unique_distributors:
            dist_lower = dist.lower()
            if dist_lower in question:
                print(f"  ✅ FOUND distributor (exact match): {dist}")
                filters["distributor"] = dist
                spec["filters"] = filters
                return spec
        
        # Try token-based matching (VERY LOOSE - any token overlap)
        best_match = None
        best_score = 0

        for dist in unique_distributors:
            dist_lower = dist.lower()
            dist_tokens = set(dist_lower.split())
            
            # Count how many dist tokens appear in question tokens
            overlap = dist_tokens & q_tokens
            score = len(overlap)
            
            if score > best_score:
                best_score = score
                best_match = dist
                if score > 0:
                    print(f"    Distributor candidate: {dist} (overlap score: {score}, tokens: {overlap})")

        if best_match and best_score > 0:
            print(f"  ✅ FOUND distributor (token match): {best_match} (score: {best_score})")
            filters["distributor"] = best_match
            spec["filters"] = filters
            return spec
        
        print(f"  ❌ No distributor match found")

    # ===== STEP 2: Fall back to matching as CUSTOMER =====
    if "Customer" not in df.columns:
        print(f"  'Customer' column not in df")
        return spec

    unique_customers = [str(x).strip() for x in df["Customer"].dropna().unique()]
    print(f"  Step 2: Checking {len(unique_customers)} unique customers...")
    
    # Try exact match first (case-insensitive)
    for cust in unique_customers:
        cust_lower = cust.lower()
        if cust_lower in question:
            print(f"  ✅ FOUND customer (exact match): {cust}")
            filters["customer"] = cust
            spec["filters"] = filters
            return spec
    
    # Try partial match
    q_tokens = [t.strip() for t in question.replace(",", " ").split() if len(t.strip()) > 2]
    best_match = None
    best_score = 0

    for cust in unique_customers:
        cust_lower = cust.lower()
        score = 0
        for tok in q_tokens:
            if tok in cust_lower or cust_lower in tok:
                score += 1
        if score > best_score:
            best_score = score
            best_match = cust
            if score > 0:
                print(f"    Customer candidate: {cust} (score: {score})")

    if best_match and best_score > 0:
        print(f"  ✅ FOUND customer (partial match): {best_match} (score: {best_score})")
        filters["customer"] = best_match
        spec["filters"] = filters
        return spec
    
    print(f"  ❌ No entity match found")
    return spec
# --------------------------------------------------------------------
# 2) EXECUTION LAYER – run the spec on the consolidated DataFrame
# --------------------------------------------------------------------

def _ensure_all_statuses_when_grouped(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the user is asking for a breakdown BY shipping status,
    we usually want to see all statuses (shipped + cancelled, etc.).
    So if shipping_status is in group_by and the mode is still the
    default 'countable', switch it to 'all'.
    """
    group_by = spec.get("group_by") or []
    ship_mode = spec.get("shipping_status_mode", "countable")

    if "shipping_status" in group_by and ship_mode in (None, "countable"):
        spec["shipping_status_mode"] = "all"

    return spec

def _force_cancellation_status_from_text(question: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override the LLM's shipping_status_mode when the question clearly says
    'cancelled by Isotopia' or 'cancelled by the customer'.
    """
    q = (question or "").lower()

    if "shipping_status_list" not in spec:
        spec["shipping_status_list"] = []

    if any(
        phrase in q
        for phrase in [
            "cancelled by isotopia",
            "canceled by isotopia",
            "rejected by isotopia",
            "cancelled by us",
            "canceled by us",
            "rejected by us",
        ]
    ):
        spec["shipping_status_mode"] = "explicit"
        spec["shipping_status_list"] = ["Rejected \\ Cancelled by Isotopia"]
        return spec

    if any(
        phrase in q
        for phrase in [
            "cancelled by the customer",
            "canceled by the customer",
            "cancelled by customer",
            "canceled by customer",
            "customer cancelled",
            "customer canceled",
            "client cancelled",
            "client canceled",
        ]
    ):
        spec["shipping_status_mode"] = "explicit"
        spec["shipping_status_list"] = ["Cancelled by the customer"]
        return spec

    return spec
def _apply_filters(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """Apply all filters from the spec to the DataFrame."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    mapping = _get_mapping(df)
    filters = spec.get("filters", {}) or {}
    
    # SAFETY: Ensure all filter values are properly typed
    if filters is None:
        filters = {}
        spec["filters"] = filters
    
    result = df.copy()

    def contains(col_name: str, value: str) -> pd.Series:
        return result[col_name].astype(str).str.contains(
            str(value), case=False, na=False, regex=False
        )

    # Customer
    if filters.get("customer") and mapping.get("customer"):
        result = result[contains(mapping["customer"], filters["customer"])]

    # --- NEW PROTECTION: avoid broken multi-entity distributor filters ---
    dist_val = filters.get("distributor")
    if isinstance(dist_val, str):
        lv = dist_val.lower()
        if (" vs " in lv or " and " in lv) and "dsd" in lv and "pi medical" in lv:
            print(f"🧹 Dropping multi-entity distributor filter: '{dist_val}'")
            filters["distributor"] = None
    # ----

    # âœ… FIXED Distributor Filter (supports single + multi-entity correctly)
    if filters.get("distributor") and mapping.get("distributor"):
        dist_val = filters["distributor"]
        col = mapping["distributor"]

        def _apply_multi_distributor(values):
            mask = None
            for v in values:
                cur = result[col].astype(str).str.contains(str(v), case=False, na=False)
                mask = cur if mask is None else (mask | cur)
            return result[mask]

        # Case 1: Proper Python list â†' ["DSD", "PI Medical"]
        if isinstance(dist_val, list):
            result = _apply_multi_distributor(dist_val)

        # Case 2: String that LOOKS like a list â†' "['DSD', 'PI Medical']"
        elif isinstance(dist_val, str):
            text = dist_val.strip()

            parsed_list = None
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, list):
                        parsed_list = parsed
                except Exception:
                    parsed_list = None

            if parsed_list:
                result = _apply_multi_distributor(parsed_list)
            else:
                # Final fallback: single distributor string
                result = result[result[col].astype(str).str.contains(text, case=False, na=False)]

        # Case 3: Any other type â†' treat as single value safely
        else:
            result = result[result[col].astype(str).str.contains(str(dist_val), case=False, na=False)]

    # Year
    if filters.get("year") and mapping.get("year"):
        result = result[result[mapping["year"]] == filters["year"]]

    # Month
    if filters.get("month") and mapping.get("month"):
        try:
            month_val = int(filters["month"])
        except Exception:
            month_val = None
        if month_val is not None:
            result = result[result[mapping["month"]] == month_val]

    # Quarter
    if filters.get("quarter") and mapping.get("quarter"):
        quarter_val = str(filters["quarter"]).upper().strip()
        if quarter_val in {"1", "2", "3", "4"}:
            quarter_val = f"Q{quarter_val}"
        result = result[result[mapping["quarter"]] == quarter_val]

    # Half-year
    if filters.get("half_year") and mapping.get("half_year"):
        hy_val = str(filters["half_year"]).upper().strip()
        if hy_val in {"1", "H1", "FIRST", "FIRST HALF", "1ST HALF"}:
            hy_val = "H1"
        elif hy_val in {"2", "H2", "SECOND", "SECOND HALF", "2ND HALF"}:
            hy_val = "H2"
        result = result[result[mapping["half_year"]] == hy_val]

    # Week (explicit single week – NOT for dynamic windows)
    if filters.get("week") and mapping.get("week"):
        result = result[result[mapping["week"]] == filters["week"]]

    # Country
    if filters.get("country") and mapping.get("country"):
        result = result[contains(mapping["country"], filters["country"])]

    # Region
    if filters.get("region") and mapping.get("region"):
        result = result[contains(mapping["region"], filters["region"])]

    # Production site
    if filters.get("production_site") and mapping.get("production_site"):
        result = result[contains(mapping["production_site"], filters["production_site"])]

    # --- Product filters (NCA / CA / Terbium aware) ---

    def _product_mask(series: pd.Series, product_query: str) -> pd.Series:
        s = series.astype(str)
        col = s.str.lower()
        q_text = str(spec.get("_question_text", ""))
        pq_text = "" if product_query is None else str(product_query)
        q = (q_text + " " + pq_text).lower()

        # Terbium / Tb-161
        if any(t in q for t in ["terbium", "tb-161", "tb161", "tb 161", " 161tb"]):
            return (
                col.str.contains("terb", na=False, regex=False)
                | col.str.contains("tb-161", na=False, regex=False)
                | col.str.contains("tb161", na=False, regex=False)
                | col.str.contains("161tb", na=False, regex=False)
            )

        # NCA (not Terbium)
        if any(
            t in q
            for t in [
                "nca",
                "n.c.a",
                "non carrier",
                "non-carrier",
                "non carrier added",
                "non-carrier added",
            ]
        ):
            nca_like = (
                col.str.contains("nca", na=False, regex=False)
                | col.str.contains("n.c.a", na=False, regex=False)
                | col.str.contains("non carrier", na=False, regex=False)
                | col.str.contains("non-carrier", na=False, regex=False)
            )
            terb_like = (
                col.str.contains("terb", na=False, regex=False)
                | col.str.contains("tb-161", na=False, regex=False)
                | col.str.contains("tb161", na=False, regex=False)
                | col.str.contains("161tb", na=False, regex=False)
            )
            return nca_like & ~terb_like

        # CA (not Terbium)
        if any(
            t in q
            for t in [
                " ca ",
                " c.a",
                " c.a.",
                "carrier added",
                " ca,",
                " ca.",
                " lu-177 ca",
                " lu 177 ca",
            ]
        ):
            ca_like = (
                col.str.contains(" c.a", na=False, regex=False)
                | col.str.contains(" c.a.", na=False, regex=False)
                | col.str.contains("carrier added", na=False, regex=False)
                | col.str.contains(" ca ", na=False, regex=False)
                | col.str.contains(" ca.", na=False, regex=False)
                | col.str.contains("(ca", na=False, regex=False)
            )
            terb_like = (
                col.str.contains("terb", na=False, regex=False)
                | col.str.contains("tb-161", na=False, regex=False)
                | col.str.contains("tb161", na=False, regex=False)
                | col.str.contains("161tb", na=False, regex=False)
            )
            return ca_like & ~terb_like

        pq = pq_text.strip().lower()
        if pq:
            return col.str.contains(pq, na=False, regex=False)

        return pd.Series(True, index=series.index)

    if filters.get("product_sold") and mapping.get("product_sold"):
        mask = _product_mask(result[mapping["product_sold"]], filters["product_sold"])
        result = result[mask]

    if filters.get("product_catalogue") and mapping.get("product_catalogue"):
        mask = _product_mask(
            result[mapping["product_catalogue"]], filters["product_catalogue"]
        )
        result = result[mask]

    # --- Shipping status filter ---
    ship_mode = spec.get("shipping_status_mode", "countable")
    ship_col = mapping.get("shipping_status")

    if ship_col:
        if ship_mode == "countable":
            result = result[result[ship_col].isin(COUNTABLE_STATUSES)]
        elif ship_mode == "cancelled":
            result = result[result[ship_col].isin(CANCELLED_STATUSES)]
        elif ship_mode == "explicit" and spec.get("shipping_status_list"):
            result = result[result[ship_col].isin(spec["shipping_status_list"])]
        elif ship_mode == "all":
            pass  # Don't filter

    # --- Dynamic week window (last N weeks logic) ---
    # CRITICAL: Only apply if time_window is properly configured
    time_window = spec.get("time_window")
    if time_window is not None and isinstance(time_window, dict):
        if time_window.get("mode"):
            result = _apply_time_window(result, spec)

    return result
def _calculate_yoy_growth(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    group_cols: List[str],
    year_col: str,
    total_col: str,
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate year-over-year growth for each entity (customer, distributor, etc.).

    Returns a table with:
      - one column per year (totals)
      - additional *_Growth columns for each consecutive year pair.
    """
    if df is None or df.empty:
        return None, float("nan")

    if not year_col or year_col not in df.columns:
        return None, float("nan")
    if not total_col or total_col not in df.columns:
        return None, float("nan")

    # Ensure numeric total column
    df = df.copy()
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)

    # Entity columns = all grouping columns except the year itself
    entity_cols = [c for c in (group_cols or []) if c and c != year_col]

    # Group to get yearly totals per entity
    group_keys = (entity_cols or []) + [year_col]
    grouped = (
        df
        .groupby(group_keys, as_index=False)[total_col]
        .sum()
    )

    # Determine which years we actually have
    years = sorted(grouped[year_col].dropna().unique().tolist())
    if len(years) < 2:
        # Not enough years to calculate YoY
        return None, float("nan")

    # Pivot: index = entity_cols (can be empty), columns = years, values = totals
    if entity_cols:
        pivot = grouped.pivot_table(
            index=entity_cols,
            columns=year_col,
            values=total_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        pivot = pivot.reset_index()
    else:
        # No entity dimension, just a single row with totals per year
        pivot = grouped.pivot_table(
            index=None,
            columns=year_col,
            values=total_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        pivot = pivot.reset_index(drop=True)

    # Normalise column names to strings
    pivot.columns = [str(c) for c in pivot.columns]

    # Add growth columns for consecutive year pairs
    for i in range(1, len(years)):
        prev_year = years[i - 1]
        curr_year = years[i]
        prev_col = str(prev_year)
        curr_col = str(curr_year)
        growth_col = f"{curr_year}_vs_{prev_year}_Growth"

        if prev_col not in pivot.columns or curr_col not in pivot.columns:
            continue

        prev_vals = pivot[prev_col].replace({0: pd.NA})
        growth = (pivot[curr_col] - pivot[prev_col]) / prev_vals * 100.0
        pivot[growth_col] = growth.round(1)

    return pivot, float("nan")

# FIXES for week-over-week (WoW) growth calculations
# These are the corrected functions to replace in query_engine.py

def _calculate_wow_growth(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    group_cols: List[str],
    week_col: str,
    total_col: str,
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate week-over-week growth for each entity.
    
    CRITICAL FIX: This returns results per (entity, week) with WoW_Growth column.
    It does NOT require a "base period" - it computes growth for ALL weeks present.
    """
    if df is None or df.empty:
        print("  âŒ _calculate_wow_growth: empty input df")
        return None, float("nan")

    if not week_col or week_col not in df.columns:
        print(f"  âŒ _calculate_wow_growth: week_col '{week_col}' not in df.columns")
        return None, float("nan")
    
    if not total_col or total_col not in df.columns:
        print(f"  âŒ _calculate_wow_growth: total_col '{total_col}' not in df.columns")
        return None, float("nan")

    df = df.copy()
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)

    # Entity columns = all grouping columns except the week itself
    entity_cols = [c for c in (group_cols or []) if c and c != week_col]
    
    print(f"  âœ… WoW: entity_cols = {entity_cols}")
    print(f"  âœ… WoW: week_col = {week_col}")
    print(f"  âœ… WoW: total_col = {total_col}")

    # Group to get weekly totals per entity
    group_keys = (entity_cols or []) + [week_col]
    print(f"  âœ… WoW: group_keys = {group_keys}")
    
    grouped = (
        df
        .groupby(group_keys, as_index=False)[total_col]
        .sum()
    )
    
    print(f"  âœ… WoW: grouped.shape after groupby = {grouped.shape}")
    print(f"  âœ… WoW: grouped.columns = {list(grouped.columns)}")

    # Sort by entity keys + week so pct_change runs in week order
    sort_keys = entity_cols + [week_col]
    grouped = grouped.sort_values(sort_keys)

    print(f"  âœ… WoW: sorted {len(grouped)} rows by {sort_keys}")

    # Compute week-over-week growth per entity
    if entity_cols:
        # For each entity, compute pct_change of total_col across weeks
        grouped["WoW_Growth_%"] = (
            grouped
            .groupby(entity_cols)[total_col]
            .pct_change() * 100.0
        )
    else:
        # No entity dimension: just compute pct_change across weeks
        grouped["WoW_Growth_%"] = grouped[total_col].pct_change() * 100.0

    # Round to 1 decimal
    grouped["WoW_Growth_%"] = grouped["WoW_Growth_%"].round(1)

    # First row of each entity will be NaN (no previous week) - keep as NaN
    
    print(f"  âœ… WoW: final grouped.shape = {grouped.shape}")
    print(f"  âœ… WoW: grouped columns = {list(grouped.columns)}")

    return grouped, float("nan")

# =========================
# Metadata relevance config
# =========================

CANCELLATION_RELEVANT_KEYWORDS = [
    "cancel", "rejected", "reject", "strike", "war", "attack",
    "spill", "maintenance", "no production", "ampoule", "problem",
    "delay", "late", "shutdown", "no ca", "no material", "low demand",
    "supply", "nrg", "houthi", "airport", "israel", "emergency"
]

DISRUPTION_KEYWORDS = [
    "spill", "late", "ampoule", "strike", "closed", "attack",
    "shutdown", "delay", "maintenance", "no production", "war",
    "weather", "airport", "shipment", "holiday", "easter", "supply",
    "problem", "issue", "shortage", "failed", "failure", "batch failed",
    "production only", "didn't supply", "didn't arrive", "forgot"
]

TREND_KEYWORDS = [
    "increase", "decrease", "demand", "no production",
    "maintenance", "delay", "problem", "shortage", "stability",
    "stable", "unstable", "drop", "spike"
]

EXCLUDE_GLOBAL_EVENTS = [
    "war in gaza",
    "war in  gaza",
    "gaza",
    "houthi",
    "airport",
    "first eu product",
    "first eu shipment",
    "first order for scantor",
    "first order for scanthor",
    "first order for scantor’s customers",
]


def _should_include_metadata(spec: Dict[str, Any]) -> bool:
    """
    Decide whether it's worth surfacing metadata at all for this question.
    """
    q = (spec.get("_question_text") or "").lower()

    if "metadata" in q or "major event" in q or "major events" in q:
        return True

    cause_keywords = [
        "why",
        "reason",
        "due to",
        "because",
        "cause",
        "caused",
        "impact",
        "influence",
        "explain",
        "explanation",
    ]
    if any(k in q for k in cause_keywords):
        return True

    disruption_keywords = [
        "cancellation",
        "cancelled",
        "canceled",
        "rejected",
        "strike",
        "war",
        "attack",
        "airport",
        "holiday",
        "easter",
        "nrg",
        "problem",
        "issue",
        "delay",
        "delayed",
        "late",
        "stability",
        "stable",
        "drop",
        "spike",
        "increase",
        "decrease",
        "change in pattern",
        "pattern change",
    ]
    if any(k in q for k in disruption_keywords):
        return True

    if spec.get("shipping_status_mode") == "cancelled":
        return True

    return False


def _build_metadata_snippet(df_filtered: pd.DataFrame, spec: Dict[str, Any]) -> Optional[str]:
    """
    Build a concise, cautious markdown snippet with contextual metadata,
    based on Meta_* columns overlapping with the filtered rows.
    """
    if df_filtered is None or df_filtered.empty:
        return None

    q = (spec.get("_question_text") or "").lower()

    df_for_metadata = df_filtered.copy()

    shipping_mode = spec.get("shipping_status_mode")
    if "cancel" in q or shipping_mode in ("cancelled", "explicit"):
        cancel_status_values = [
            "Rejected \\ Cancelled by Isotopia",
            "Cancelled by the customer",
            "Rejected \\ Cancelled",
        ]
        ship_col = None
        for c in df_filtered.columns:
            lc = str(c).lower().replace(" ", "")
            if lc in ("shippingstatus", "shipping_status"):
                ship_col = c
                break

        if ship_col is not None:
            df_for_metadata = df_filtered[df_filtered[ship_col].isin(cancel_status_values)]
            if df_for_metadata.empty:
                return None

    meta_cols = [
        c for c in df_for_metadata.columns
        if c.startswith("Meta_") and pd.api.types.is_object_dtype(df_for_metadata[c])
    ]
    if not meta_cols:
        return None

# 🔧 FIX: For "why" questions, show ALL metadata for the filtered week/period
    # rather than requiring keyword matches
    is_why_question = any(k in q for k in ["why", "reason", "drop", "increase", "decrease", "cause"])
    
    if "cancel" in q or "rejected" in q:
        relevant_keys = CANCELLATION_RELEVANT_KEYWORDS
    elif any(k in q for k in ["why", "impact", "drop", "increase", "decrease", "change", "effect"]):
        relevant_keys = DISRUPTION_KEYWORDS
    else:
        relevant_keys = TREND_KEYWORDS

    unique_events: List[str] = []

    for col in meta_cols:
        series = df_for_metadata[col].dropna().astype(str).str.strip()
        series = series[series != ""]
        if series.empty:
            continue

        for raw_val in series.unique():
            for part in str(raw_val).split(";"):
                event = part.strip()
                if not event:
                    continue

                ev_lower = event.lower()

                if any(ex in ev_lower for ex in EXCLUDE_GLOBAL_EVENTS):
                    continue

                # 🔧 FIX: For "why" questions, include ALL events for that week
                # Otherwise, use keyword filtering
                if is_why_question:
                    # Show everything for the specific week/period
                    if event not in unique_events:
                        unique_events.append(event)
                else:
                    # Normal filtering for non-"why" questions
                    if not any(key in ev_lower for key in relevant_keys):
                        continue
                    if event not in unique_events:
                        unique_events.append(event)

    if not unique_events:
        return None

    max_events = 8
    events_to_show = unique_events[:max_events]

    years = []
    weeks = []
    if "Year" in df_for_metadata.columns:
        years = sorted({int(y) for y in df_for_metadata["Year"].dropna().unique()})
    if "Week" in df_for_metadata.columns:
        weeks = sorted({int(w) for w in df_for_metadata["Week"].dropna().unique()})

    scope_bits = []
    if years:
        if len(years) == 1:
            scope_bits.append(f"year {years[0]}")
        else:
            scope_bits.append("years " + ", ".join(str(y) for y in years))
    if weeks:
        if len(weeks) <= 4:
            scope_bits.append("weeks " + ", ".join(str(w) for w in weeks))
        else:
            scope_bits.append(f"weeks {weeks[0]}–{weeks[-1]}")

    if scope_bits:
        scope_text = ", ".join(scope_bits)
    else:
        scope_text = "the period covered by these orders"

    intro = (
        f"\n\n**Contextual events overlapping with {scope_text}** "
        "(these may help explain patterns, but they do **not** prove a direct cause-and-effect):\n"
    )

    bullets = "\n".join(f"- {e}" for e in events_to_show)

    if len(unique_events) > max_events:
        bullets += "\n- … (additional events exist for these weeks but are not shown here)"

    return intro + bullets

# ============================================================================
# COMPLETE FIXED FUNCTIONS FOR PROJECTION VS ACTUAL
# ============================================================================

def _run_aggregation(
    df_filtered: pd.DataFrame,
    spec: Dict[str, Any],
    full_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Run the specified aggregation (sum, average, share_of_total, growth_rate, top_n, compare, projection_vs_actual)
    on the filtered data.
    """
    if df_filtered is None or not isinstance(df_filtered, pd.DataFrame) or df_filtered.empty:
        return None, float("nan")

    base_df = df_filtered
    mapping = _get_mapping(base_df)
    total_col = mapping.get("total_mci")

    if not total_col or total_col not in base_df.columns:
        return None, float("nan")

    aggregation = spec.get("aggregation", "sum_mci") or "sum_mci"
    group_by = spec.get("group_by") or []
    group_cols = [mapping.get(field) for field in group_by if mapping.get(field)]

    print(f"\n{'='*70}")
    print(f"_run_aggregation() - {aggregation.upper()}")
    print(f"{'='*70}")
    print(f"  Input shape: {base_df.shape}")
    print(f"  Aggregation: {aggregation}")
    print(f"  Group by: {group_by}")
    print(f"  Total col: {total_col}")

    # ===========================================================================
    # PROJECTION_VS_ACTUAL MODE - HANDLE FIRST
    # ===========================================================================
    if aggregation == "projection_vs_actual":
        print(f"\n[ROUTING] Detected projection_vs_actual - calling handler")
        return _run_projection_vs_actual_aggregation(base_df, spec, mapping, total_col)

    # ===========================================================================
    # COMPARE MODE (entity or time comparison)
    # ===========================================================================
    if aggregation == "compare":
        return _run_compare_aggregation(base_df, spec, mapping, total_col)

    # ===========================================================================
    # TOP N (ranking entities by volume)
    # ===========================================================================
    if aggregation == "top_n":
        return _run_top_n_aggregation(base_df, spec, mapping, total_col)

    # ===========================================================================
    # GROWTH RATE (WoW / YoY)
    # ===========================================================================
    if aggregation == "growth_rate":
        return _run_growth_rate_aggregation(base_df, spec, mapping, total_col, group_cols)

    # ===========================================================================
    # STANDARD AGGREGATIONS (sum, average, share_of_total)
    # ===========================================================================
    
    # For sum_mci: just sum the total_col
    if aggregation == "sum_mci":
        overall_value = float(base_df[total_col].sum())
        
        if not group_cols:
            # No grouping: just return the sum
            print(f"  No grouping → returning scalar: {overall_value:.0f}")
            return None, overall_value
        
        # Group and sum
        grouped_df = base_df.groupby(group_cols, as_index=False)[total_col].sum()
        grouped_df = grouped_df.sort_values(total_col, ascending=False)
        
        # Format mCi column
        grouped_df[total_col] = grouped_df[total_col].round(0).astype(int)
        
        print(f"  Grouped result shape: {grouped_df.shape}")
        return grouped_df, overall_value

    # For average_mci
    if aggregation == "average_mci":
        overall_value = float(base_df[total_col].mean())
        
        if not group_cols:
            print(f"  No grouping → returning scalar: {overall_value:.0f}")
            return None, overall_value
        
        # Group and average
        grouped_df = base_df.groupby(group_cols, as_index=False)[total_col].mean()
        grouped_df = grouped_df.sort_values(total_col, ascending=False)
        grouped_df[total_col] = grouped_df[total_col].round(0).astype(int)
        
        print(f"  Grouped result shape: {grouped_df.shape}")
        return grouped_df, overall_value

    # For share_of_total
    if aggregation == "share_of_total":
        total_overall = float(base_df[total_col].sum())
        
        if not group_cols:
            # Single value: share is 100%
            overall_value = 1.0
            print(f"  No grouping → returning scalar: {overall_value*100:.1f}%")
            return None, overall_value
        
        # Group and calculate share
        grouped_df = base_df.groupby(group_cols, as_index=False)[total_col].sum()
        grouped_df["Share_%"] = (grouped_df[total_col] / total_overall * 100.0).round(1)
        grouped_df = grouped_df.sort_values(total_col, ascending=False)
        
        grouped_df[total_col] = grouped_df[total_col].round(0).astype(int)
        
        # Store denominator for display
        share_debug = {"numerator": float(grouped_df[total_col].sum()), "denominator": total_overall}
        spec["_share_debug"] = share_debug
        
        print(f"  Grouped result shape: {grouped_df.shape}")
        return grouped_df, (grouped_df[total_col].sum() / total_overall)

    # Fallback: unknown aggregation
    print(f"  ⚠️ Unknown aggregation type: {aggregation}")
    overall_value = float(base_df[total_col].sum())
    return None, overall_value

# ===========================================================================
# HELPER: COMPARE AGGREGATION
# ===========================================================================
def _run_compare_aggregation(
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    mapping: Dict[str, Optional[str]],
    total_col: str,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Execute COMPARE aggregation (side-by-side entity comparison).
    Handles: distributors, products, customers, etc.
    """
    compare = spec.get("compare") or {}
    if compare is None:
        compare = {}
    spec["compare"] = compare

    entities = compare.get("entities")
    if entities is None:
        entities = []
    compare["entities"] = entities
    spec["compare"] = compare

    entity_type = compare.get("entity_type")

    print(f"\n[COMPARE] Starting compare mode")
    print(f"  entities: {entities}")
    print(f"  entity_type: {entity_type}")

    # =========================================================================
    # DISTRIBUTOR COMPARISON
    # =========================================================================
    if entities and entity_type == "distributor":
        entity_col = mapping.get("distributor")
        print(f"\n[COMPARE-DIST] Distributor comparison")
        print(f"  entity_col: {entity_col}")
        print(f"  base_df.shape: {base_df.shape}")
        print(f"  entities to compare: {entities}")
        
        # DEBUG: Show what's actually in the distributor column
        if entity_col and entity_col in base_df.columns:
            unique_dists = base_df[entity_col].dropna().unique()
            print(f"  Unique distributors in base_df: {list(unique_dists)[:10]}")

        if not entity_col or entity_col not in base_df.columns:
            print(f"  ❌ ERROR: entity_col '{entity_col}' not in base_df.columns")
            print(f"  Available columns: {list(base_df.columns)[:15]}")
            return None, float("nan")

        # Build comparison data for EACH entity
        comparison_data = []

        for ent in entities:
            ent_lower = str(ent).lower().strip()
            print(f"\n  Searching for: '{ent}' (normalized: '{ent_lower}')")

            # Case-insensitive substring match
            mask = base_df[entity_col].astype(str).str.lower().str.contains(
                ent_lower, case=False, na=False, regex=False
            )
            sub = base_df[mask]

            print(f"    Found {len(sub)} matching rows")
            
            # DEBUG: Show sample values that matched
            if len(sub) > 0:
                sample_dists = sub[entity_col].unique()[:3]
                print(f"    Sample matches: {sample_dists}")

            if len(sub) > 0:
                total_mci = float(sub[total_col].sum())
                count = len(sub)
                avg_mci = total_mci / count if count > 0 else 0

                comparison_data.append({
                    "Distributor": ent,
                    "Total_mCi": int(round(total_mci)),
                    "Count": count,
                    "Average_mCi": int(round(avg_mci))
                })
                print(f"    ✅ Added: {int(round(total_mci))} mCi, {count} orders")
            else:
                # Add zero row for missing entity
                comparison_data.append({
                    "Distributor": ent,
                    "Total_mCi": 0,
                    "Count": 0,
                    "Average_mCi": 0
                })
                print(f"    ⚠️  No data - added zero row")
                # DEBUG: Show what values are actually in the column
                all_vals = base_df[entity_col].dropna().unique()
                print(f"    All distributor values in base_df:")
                for val in all_vals[:15]:
                    print(f"      - '{val}'")

        if not comparison_data:
            print(f"  ❌ No comparison data generated!")
            return None, float("nan")

        # Build result DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        total_mci = float(comparison_df["Total_mCi"].sum())

        print(f"\n[COMPARE-DIST] ✅ Complete!")
        print(f"  Result shape: {comparison_df.shape}")
        print(f"  Total mCi: {total_mci:.0f}")

        return comparison_df, total_mci

    # =========================================================================
    # PRODUCT COMPARISON
    # =========================================================================
    if entities and entity_type in ("product_sold", "product_catalogue"):
        prod_col = (
            mapping.get("product_sold")
            if entity_type == "product_sold"
            else mapping.get("product_catalogue")
        )

        print(f"\n[COMPARE-PROD] Product comparison")
        print(f"  entity_type: {entity_type}")
        print(f"  prod_col: {prod_col}")

        if not prod_col or prod_col not in base_df.columns:
            print(f"  ❌ ERROR: prod_col '{prod_col}' not in base_df.columns")
            return None, float("nan")

        def _product_subset(df: pd.DataFrame, label: str) -> pd.DataFrame:
            """Filter df to rows matching product label (NCA/CA/Terbium/generic)."""
            label_lower = str(label).lower().strip()
            series = df[prod_col].astype(str).str.lower()

            nca_like = (
                series.str.contains("n.c.a", regex=False, na=False)
                | series.str.contains(" nca", regex=False, na=False)
                | series.str.contains("non carrier", regex=False, na=False)
                | series.str.contains("non-carrier", regex=False, na=False)
            )

            ca_like_raw = (
                series.str.contains(" c.a", regex=False, na=False)
                | series.str.contains("(c.a", regex=False, na=False)
                | series.str.contains(" c.a.", regex=False, na=False)
                | series.str.contains(" ca ", regex=False, na=False)
                | series.str.contains(" carrier added", regex=False, na=False)
            )

            terb_like = (
                series.str.contains("terb", regex=False, na=False)
                | series.str.contains("tb-161", regex=False, na=False)
                | series.str.contains("tb161", regex=False, na=False)
                | series.str.contains("161tb", regex=False, na=False)
            )

            if label_lower == "nca":
                return df[nca_like & ~terb_like]

            if label_lower == "ca":
                return df[ca_like_raw & ~nca_like & ~terb_like]

            if label_lower in ("terbium", "tb-161", "tb161", "161tb"):
                return df[terb_like]

            # Generic substring match
            return df[series.str.contains(label_lower, na=False, regex=False)]

        # Build comparison
        comparison_data = []

        for ent in entities:
            sub = _product_subset(base_df, ent)

            if sub.empty:
                comparison_data.append({
                    "Product": ent.upper(),
                    "Total_mCi": 0,
                    "Count": 0,
                    "Average_mCi": 0
                })
                continue

            total_mci = float(sub[total_col].sum())
            count = len(sub)
            avg_mci = total_mci / count if count > 0 else 0

            comparison_data.append({
                "Product": ent.upper(),
                "Total_mCi": int(round(total_mci)),
                "Count": count,
                "Average_mCi": int(round(avg_mci))
            })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            total_mci = float(comparison_df["Total_mCi"].sum())
            print(f"[COMPARE-PROD] ✅ Complete! Total: {total_mci:.0f}")
            return comparison_df, total_mci

        return None, float("nan")

    # =========================================================================
    # CUSTOMER COMPARISON (similar to distributor)
    # =========================================================================
    if entities and entity_type == "customer":
        entity_col = mapping.get("customer")

        print(f"\n[COMPARE-CUST] Customer comparison")
        print(f"  entity_col: {entity_col}")

        if not entity_col or entity_col not in base_df.columns:
            print(f"  ❌ ERROR: entity_col '{entity_col}' not in base_df.columns")
            return None, float("nan")

        comparison_data = []

        for ent in entities:
            ent_lower = str(ent).lower().strip()

            mask = base_df[entity_col].astype(str).str.lower().str.contains(
                ent_lower, case=False, na=False, regex=False
            )
            sub = base_df[mask]

            if len(sub) > 0:
                total_mci = float(sub[total_col].sum())
                count = len(sub)
                avg_mci = total_mci / count if count > 0 else 0

                comparison_data.append({
                    "Customer": ent,
                    "Total_mCi": int(round(total_mci)),
                    "Count": count,
                    "Average_mCi": int(round(avg_mci))
                })
            else:
                comparison_data.append({
                    "Customer": ent,
                    "Total_mCi": 0,
                    "Count": 0,
                    "Average_mCi": 0
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            total_mci = float(comparison_df["Total_mCi"].sum())
            print(f"[COMPARE-CUST] ✅ Complete! Total: {total_mci:.0f}")
            return comparison_df, total_mci

        return None, float("nan")

    # =========================================================================
    # FALLBACK: Simple grouping if no entity type matched
    # =========================================================================
    print(f"[COMPARE] No specific entity type matched, falling back to grouping")

    if not group_cols:
        total_mci = float(base_df[total_col].sum())
        return None, total_mci

    grouped_df = base_df.groupby(group_cols, as_index=False)[total_col].sum()
    grouped_df[total_col] = grouped_df[total_col].round(0).astype("int64")
    total_mci = float(grouped_df[total_col].sum())

    print(f"[COMPARE] Fallback grouping: shape={grouped_df.shape}, total={total_mci:.0f}")
    return grouped_df, total_mci


# ===========================================================================
# HELPER: TOP N AGGREGATION
# ===========================================================================
def _run_top_n_aggregation(
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    mapping: Dict[str, Optional[str]],
    total_col: str,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Execute TOP N aggregation (rank entities by volume).
    """
    n_value = spec.get("_top_n_value", 10)
    rank_entity = spec.get("_top_n_entity")
    group_by = spec.get("group_by") or []

    print(f"\n[TOP_N]")
    print(f"  n_value: {n_value}")
    print(f"  rank_entity: {rank_entity}")
    print(f"  group_by: {group_by}")

    if not rank_entity:
        rank_entity = group_by[0] if group_by else None

    if not rank_entity:
        print(f"  ❌ No entity to rank on")
        return None, float("nan")

    entity_col = mapping.get(rank_entity)

    if not entity_col or entity_col not in base_df.columns:
        print(f"  ❌ entity_col '{entity_col}' not in base_df.columns")
        return None, float("nan")

    # Group by entity and sum totals
    group_cols = [entity_col]
    grouped_df = base_df.groupby(group_cols, as_index=False)[total_col].sum()

    # Sort by total descending and take top N
    grouped_df = grouped_df.sort_values(total_col, ascending=False)
    top_df = grouped_df.head(n_value).reset_index(drop=True)

    # Add rank column
    top_df.insert(0, "Rank", range(1, len(top_df) + 1))

    # Format mCi column
    top_df[total_col] = top_df[total_col].round(0).astype(int)

    total_volume = float(base_df[total_col].sum())

    print(f"  ✅ Top {n_value} result: shape={top_df.shape}, total={total_volume:.0f}")
    return top_df, total_volume


# ===========================================================================
# HELPER: GROWTH RATE AGGREGATION (WoW / YoY)
# ===========================================================================
def _run_growth_rate_aggregation(
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    mapping: Dict[str, Optional[str]],
    total_col: str,
    group_cols: List[str],
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Execute GROWTH RATE aggregation.
    Routes to WoW or YoY depending on group_by and time_window.
    """
    group_by = spec.get("group_by") or []
    time_window = spec.get("time_window") or {}
    compare = spec.get("compare") or {}

    print(f"\n[GROWTH_RATE]")
    print(f"  time_window.mode: {time_window.get('mode')}")
    print(f"  compare.entities: {compare.get('entities')}")
    print(f"  compare.entity_type: {compare.get('entity_type')}")
    print(f"  group_by: {group_by}")

    entities = compare.get("entities")
    if entities is None:
        entities = []
    entity_type = compare.get("entity_type")

    # =========================================================================
    # STEP 1: Filter by specific entities if comparing
    # =========================================================================
    if entities and entity_type:
        entity_col = mapping.get(entity_type)
        print(f"  Filtering by entities: {entities}")

        if entity_col and entity_col in base_df.columns:
            mask = None
            for ent in entities:
                cur = base_df[entity_col].astype(str).str.contains(
                    str(ent), case=False, na=False, regex=False
                )
                if mask is None:
                    mask = cur
                else:
                    mask = mask | cur

            if mask is not None:
                base_df = base_df[mask]
                print(f"  After entity filtering: {len(base_df)} rows")

    # =========================================================================
    # STEP 2: Determine growth type (WoW vs YoY)
    # =========================================================================

    # Case 1: Dynamic week window → WoW
    if time_window.get("mode") in ("last_n_weeks", "anchored_last_n_weeks"):
        week_col = mapping.get("week")
        if week_col and week_col in base_df.columns:
            print(f"  → Case 1: Dynamic week window (WoW)")
            group_df, overall_val = _calculate_wow_growth(
                base_df, spec, group_cols, week_col, total_col
            )
            return group_df, overall_val

    # Case 2: Explicit weekly grouping → WoW
    week_col = mapping.get("week")
    if "week" in group_by and week_col and week_col in base_df.columns:
        print(f"  → Case 2: Week in group_by (WoW)")
        group_df, overall_val = _calculate_wow_growth(
            base_df, spec, group_cols, week_col, total_col
        )
        # Pivot by entity if comparing
        if entities and entity_type:
            group_df = _pivot_growth_by_entity(group_df, spec)
        return group_df, overall_val

    # Case 3: Year-over-year grouping → YoY
    year_col = mapping.get("year")
    if "year" in group_by and year_col and year_col in base_df.columns:
        print(f"  → Case 3: Year in group_by (YoY)")
        group_df, overall_val = _calculate_yoy_growth(
            base_df, spec, group_cols, year_col, total_col
        )
        return group_df, overall_val

    # Fallback: cannot compute growth
    print(f"  ⚠️ No growth calculation possible")
    return None, float("nan")
# ===========================================================================
# HELPER: COMPARE AGGREGATION
# ===========================================================================
def _run_compare_aggregation(
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    mapping: Dict[str, Optional[str]],
    total_col: str,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Execute COMPARE aggregation (side-by-side entity comparison).
    Handles: distributors, products, customers, etc.
    """
    compare = spec.get("compare") or {}
    if compare is None:
        compare = {}
    spec["compare"] = compare

    entities = compare.get("entities")
    if entities is None:
        entities = []
    compare["entities"] = entities
    spec["compare"] = compare

    entity_type = compare.get("entity_type")

    print(f"\n[COMPARE] Starting compare mode")
    print(f"  entities: {entities}")
    print(f"  entity_type: {entity_type}")

    # =========================================================================
    # DISTRIBUTOR COMPARISON
    # =========================================================================
    if entities and entity_type == "distributor":
        entity_col = mapping.get("distributor")
        print(f"\n[COMPARE-DIST] Distributor comparison")
        print(f"  entity_col: {entity_col}")
        print(f"  base_df.shape: {base_df.shape}")
        print(f"  entities to compare: {entities}")
        
        # DEBUG: Show what's actually in the distributor column
        if entity_col and entity_col in base_df.columns:
            unique_dists = base_df[entity_col].dropna().unique()
            print(f"  Unique distributors in base_df: {list(unique_dists)[:10]}")

        if not entity_col or entity_col not in base_df.columns:
            print(f"  ❌ ERROR: entity_col '{entity_col}' not in base_df.columns")
            print(f"  Available columns: {list(base_df.columns)[:15]}")
            return None, float("nan")

        # Build comparison data for EACH entity
        comparison_data = []

        for ent in entities:
            ent_lower = str(ent).lower().strip()
            print(f"\n  Searching for: '{ent}' (normalized: '{ent_lower}')")

            # Case-insensitive substring match
            mask = base_df[entity_col].astype(str).str.lower().str.contains(
                ent_lower, case=False, na=False, regex=False
            )
            sub = base_df[mask]

            print(f"    Found {len(sub)} matching rows")
            
            # DEBUG: Show sample values that matched
            if len(sub) > 0:
                sample_dists = sub[entity_col].unique()[:3]
                print(f"    Sample matches: {sample_dists}")

            if len(sub) > 0:
                total_mci = float(sub[total_col].sum())
                count = len(sub)
                avg_mci = total_mci / count if count > 0 else 0

                comparison_data.append({
                    "Distributor": ent,
                    "Total_mCi": int(round(total_mci)),
                    "Count": count,
                    "Average_mCi": int(round(avg_mci))
                })
                print(f"    ✅ Added: {int(round(total_mci))} mCi, {count} orders")
            else:
                # Add zero row for missing entity
                comparison_data.append({
                    "Distributor": ent,
                    "Total_mCi": 0,
                    "Count": 0,
                    "Average_mCi": 0
                })
                print(f"    ⚠️  No data - added zero row")
                # DEBUG: Show what values are actually in the column
                all_vals = base_df[entity_col].dropna().unique()
                print(f"    All distributor values in base_df:")
                for val in all_vals[:15]:
                    print(f"      - '{val}'")

        if not comparison_data:
            print(f"  ❌ No comparison data generated!")
            return None, float("nan")

        # Build result DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        total_mci = float(comparison_df["Total_mCi"].sum())

        print(f"\n[COMPARE-DIST] ✅ Complete!")
        print(f"  Result shape: {comparison_df.shape}")
        print(f"  Total mCi: {total_mci:.0f}")

        return comparison_df, total_mci

    # =========================================================================
    # PRODUCT COMPARISON
    # =========================================================================
    if entities and entity_type in ("product_sold", "product_catalogue"):
        prod_col = (
            mapping.get("product_sold")
            if entity_type == "product_sold"
            else mapping.get("product_catalogue")
        )

        print(f"\n[COMPARE-PROD] Product comparison")
        print(f"  entity_type: {entity_type}")
        print(f"  prod_col: {prod_col}")

        if not prod_col or prod_col not in base_df.columns:
            print(f"  ❌ ERROR: prod_col '{prod_col}' not in base_df.columns")
            return None, float("nan")

        def _product_subset(df: pd.DataFrame, label: str) -> pd.DataFrame:
            """Filter df to rows matching product label (NCA/CA/Terbium/generic)."""
            label_lower = str(label).lower().strip()
            series = df[prod_col].astype(str).str.lower()

            nca_like = (
                series.str.contains("n.c.a", regex=False, na=False)
                | series.str.contains(" nca", regex=False, na=False)
                | series.str.contains("non carrier", regex=False, na=False)
                | series.str.contains("non-carrier", regex=False, na=False)
            )

            ca_like_raw = (
                series.str.contains(" c.a", regex=False, na=False)
                | series.str.contains("(c.a", regex=False, na=False)
                | series.str.contains(" c.a.", regex=False, na=False)
                | series.str.contains(" ca ", regex=False, na=False)
                | series.str.contains(" carrier added", regex=False, na=False)
            )

            terb_like = (
                series.str.contains("terb", regex=False, na=False)
                | series.str.contains("tb-161", regex=False, na=False)
                | series.str.contains("tb161", regex=False, na=False)
                | series.str.contains("161tb", regex=False, na=False)
            )

            if label_lower == "nca":
                return df[nca_like & ~terb_like]

            if label_lower == "ca":
                return df[ca_like_raw & ~nca_like & ~terb_like]

            if label_lower in ("terbium", "tb-161", "tb161", "161tb"):
                return df[terb_like]

            # Generic substring match
            return df[series.str.contains(label_lower, na=False, regex=False)]

        # Build comparison
        comparison_data = []

        for ent in entities:
            sub = _product_subset(base_df, ent)

            if sub.empty:
                comparison_data.append({
                    "Product": ent.upper(),
                    "Total_mCi": 0,
                    "Count": 0,
                    "Average_mCi": 0
                })
                continue

            total_mci = float(sub[total_col].sum())
            count = len(sub)
            avg_mci = total_mci / count if count > 0 else 0

            comparison_data.append({
                "Product": ent.upper(),
                "Total_mCi": int(round(total_mci)),
                "Count": count,
                "Average_mCi": int(round(avg_mci))
            })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            total_mci = float(comparison_df["Total_mCi"].sum())
            print(f"[COMPARE-PROD] ✅ Complete! Total: {total_mci:.0f}")
            return comparison_df, total_mci

        return None, float("nan")

    # =========================================================================
    # CUSTOMER COMPARISON (similar to distributor)
    # =========================================================================
    if entities and entity_type == "customer":
        entity_col = mapping.get("customer")

        print(f"\n[COMPARE-CUST] Customer comparison")
        print(f"  entity_col: {entity_col}")

        if not entity_col or entity_col not in base_df.columns:
            print(f"  ❌ ERROR: entity_col '{entity_col}' not in base_df.columns")
            return None, float("nan")

        comparison_data = []

        for ent in entities:
            ent_lower = str(ent).lower().strip()

            mask = base_df[entity_col].astype(str).str.lower().str.contains(
                ent_lower, case=False, na=False, regex=False
            )
            sub = base_df[mask]

            if len(sub) > 0:
                total_mci = float(sub[total_col].sum())
                count = len(sub)
                avg_mci = total_mci / count if count > 0 else 0

                comparison_data.append({
                    "Customer": ent,
                    "Total_mCi": int(round(total_mci)),
                    "Count": count,
                    "Average_mCi": int(round(avg_mci))
                })
            else:
                comparison_data.append({
                    "Customer": ent,
                    "Total_mCi": 0,
                    "Count": 0,
                    "Average_mCi": 0
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            total_mci = float(comparison_df["Total_mCi"].sum())
            print(f"[COMPARE-CUST] ✅ Complete! Total: {total_mci:.0f}")
            return comparison_df, total_mci

        return None, float("nan")

    # =========================================================================
    # FALLBACK: Simple grouping if no entity type matched
    # =========================================================================
    print(f"[COMPARE] No specific entity type matched, falling back to grouping")

    if not group_cols:
        total_mci = float(base_df[total_col].sum())
        return None, total_mci

    grouped_df = base_df.groupby(group_cols, as_index=False)[total_col].sum()
    grouped_df[total_col] = grouped_df[total_col].round(0).astype("int64")
    total_mci = float(grouped_df[total_col].sum())

    print(f"[COMPARE] Fallback grouping: shape={grouped_df.shape}, total={total_mci:.0f}")
    return grouped_df, total_mci


# ===========================================================================
# HELPER: TOP N AGGREGATION
# ===========================================================================
def _run_top_n_aggregation(
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    mapping: Dict[str, Optional[str]],
    total_col: str,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Execute TOP N aggregation (rank entities by volume).
    """
    n_value = spec.get("_top_n_value", 10)
    rank_entity = spec.get("_top_n_entity")
    group_by = spec.get("group_by") or []

    print(f"\n[TOP_N]")
    print(f"  n_value: {n_value}")
    print(f"  rank_entity: {rank_entity}")
    print(f"  group_by: {group_by}")

    if not rank_entity:
        rank_entity = group_by[0] if group_by else None

    if not rank_entity:
        print(f"  ❌ No entity to rank on")
        return None, float("nan")

    entity_col = mapping.get(rank_entity)

    if not entity_col or entity_col not in base_df.columns:
        print(f"  ❌ entity_col '{entity_col}' not in base_df.columns")
        return None, float("nan")

    # Group by entity and sum totals
    group_cols = [entity_col]
    grouped_df = base_df.groupby(group_cols, as_index=False)[total_col].sum()

    # Sort by total descending and take top N
    grouped_df = grouped_df.sort_values(total_col, ascending=False)
    top_df = grouped_df.head(n_value).reset_index(drop=True)

    # Add rank column
    top_df.insert(0, "Rank", range(1, len(top_df) + 1))

    # Format mCi column
    top_df[total_col] = top_df[total_col].round(0).astype(int)

    total_volume = float(base_df[total_col].sum())

    print(f"  ✅ Top {n_value} result: shape={top_df.shape}, total={total_volume:.0f}")
    return top_df, total_volume


# ===========================================================================
# HELPER: GROWTH RATE AGGREGATION (WoW / YoY)
# ===========================================================================
def _run_growth_rate_aggregation(
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    mapping: Dict[str, Optional[str]],
    total_col: str,
    group_cols: List[str],
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Growth rate over time. Supports:
      - Weekly (default)
      - Monthly (YearMonth)
      - Quarterly (YearQuarter)

    Output columns:
      Period | Actual | Prev_Actual | Growth_%
    """
    if base_df is None or base_df.empty:
        return None, float("nan")

    df = base_df.copy()

    actual_col = mapping.get("total_mci") or total_col
    if not actual_col or actual_col not in df.columns:
        return None, float("nan")

    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce").fillna(0.0)

    # Identify year/week columns
    year_col = "Year" if "Year" in df.columns else mapping.get("year")
    week_col = mapping.get("week")

    # If your mapping uses the production week field name, week_col will point to it.
    # Otherwise fallback to the common column name you use elsewhere.
    if not week_col or week_col not in df.columns:
        if "Week number for Activity vs Projection" in df.columns:
            week_col = "Week number for Activity vs Projection"

    # Determine requested period from group_by
    group_by = [str(x).lower() for x in (spec.get("group_by") or [])]
    wants_month = any(x in ["month", "monthly"] for x in group_by)
    wants_quarter = any(x in ["quarter", "quarterly", "q"] for x in group_by)

    # Default is weekly unless month/quarter explicitly requested
    period_mode = "week"
    if wants_month:
        period_mode = "month"
    elif wants_quarter:
        period_mode = "quarter"

    # Derive period columns if needed
    if period_mode in ["month", "quarter"]:
        if not year_col or year_col not in df.columns or not week_col or week_col not in df.columns:
            # Cannot derive month/quarter without year+week
            return None, float("nan")
        df = _ensure_period_columns(df, year_col=year_col, week_col=week_col)

    # Choose the period column
    if period_mode == "week":
        # Use explicit group col if provided; else default to week column
        period_col = group_cols[0] if group_cols else week_col
        if not period_col or period_col not in df.columns:
            return None, float("nan")
        period_label = "Week"
    elif period_mode == "month":
        period_col = "YearMonth"
        period_label = "Month"
    else:  # quarter
        period_col = "YearQuarter"
        period_label = "Quarter"

    # Aggregate
    series = (
        df.groupby(period_col, as_index=False)[actual_col]
          .sum()
          .rename(columns={actual_col: "Actual", period_col: period_label})
    )

    # Sort periods properly
    if period_mode == "week":
        series[period_label] = pd.to_numeric(series[period_label], errors="coerce")
        series = series.dropna(subset=[period_label]).sort_values(period_label)
        series[period_label] = series[period_label].astype(int)
    elif period_mode == "month":
        # YYYY-MM sorts lexicographically correctly
        series = series.sort_values(period_label)
    else:
        # YYYY-Qn sorts lexicographically correctly if format is consistent
        series = series.sort_values(period_label)

    if len(series) < 2:
        return None, float("nan")

    series["Prev_Actual"] = series["Actual"].shift(1)

    def _growth(curr, prev):
        if pd.isna(prev) or prev == 0:
            return float("inf") if curr > 0 else 0.0
        return (curr - prev) / prev * 100.0

    series["Growth_%"] = [
        _growth(c, p) for c, p in zip(series["Actual"].tolist(), series["Prev_Actual"].tolist())
    ]
    series["Growth_%"] = pd.to_numeric(series["Growth_%"], errors="coerce").round(1)

    # Format numeric columns
    series["Actual"] = series["Actual"].round(0).astype(int)
    series["Prev_Actual"] = series["Prev_Actual"].fillna(0).round(0).astype(int)

    total_actual = float(series["Actual"].sum())
    return series, total_actual

def _build_chart_block(
    group_df: pd.DataFrame,
    spec: Dict[str, Any],
    aggregation: str
) -> Optional[str]:
    """Build a chart specification for Altair visualization."""
    import json

    if group_df is None or group_df.empty:
        return None

    # ---- Prepare data for charting ----
    chart_data = group_df.copy()
    cols = list(chart_data.columns)

    if not cols:
        return None

    group_by = spec.get("group_by") or []

    # ✅ SAFE question text handling (FIX)
    q_text = spec.get("_question_text")
    q_lower = str(q_text).lower() if q_text is not None else ""

    # ---- Special handling: combine Year + Week into a time identifier ----
    if "Year" in chart_data.columns and "Week" in chart_data.columns:
        chart_data["YearWeek"] = (
            chart_data["Year"].astype(str)
            + "-W"
            + chart_data["Week"].astype(str).str.zfill(2)
        )

        if "Year" not in group_by or len(group_by) > 1:
            cols = [c for c in cols if c != "Year"]
            cols.insert(0, "YearWeek")
            chart_data = chart_data[
                cols + [c for c in chart_data.columns if c not in cols]
            ]

    cols = list(chart_data.columns)

    # ---- Determine chart type ----
    chart_type = "bar"

    if any(w in q_lower for w in ["line", "trend", "over time", "week"]):
        chart_type = "line"
    elif any(w in q_lower for w in ["pie", "share", "percentage"]):
        chart_type = "pie"

    # ---- Infer X field (category / time) ----
    x_field = None
    for col in cols:
        if not pd.api.types.is_numeric_dtype(chart_data[col]):
            x_field = col
            break

    if x_field is None:
        x_field = cols[0]

    # ---- Infer Y field (metric) ----
    priority_keywords = [
        "mci", "growth", "delta", "pct", "average", "actual", "projected"
    ]

    y_field = None

    # First pass: priority metric columns
    for col in cols:
        if pd.api.types.is_numeric_dtype(chart_data[col]):
            col_lower = str(col).lower()  # ✅ safe
            if any(kw in col_lower for kw in priority_keywords):
                y_field = col
                break

    # Fallback: any numeric column excluding Week/Year
    if y_field is None:
        for col in cols:
            if (
                pd.api.types.is_numeric_dtype(chart_data[col])
                and col not in ("Week", "Year")
            ):
                y_field = col
                break

    if y_field is None:
        if len(cols) > 1:
            y_field = cols[1]
        else:
            return None

    # ---- Series field (color grouping) ----
    series_field = None
    if chart_type in ("line", "bar"):
        for col in cols:
            if (
                col != x_field
                and col != y_field
                and not pd.api.types.is_numeric_dtype(chart_data[col])
            ):
                series_field = col
                break

    # ---- Build chart spec ----
    spec_dict = {
        "type": chart_type,
        "xField": x_field,
        "yField": y_field,
        "seriesField": series_field,
        "aggregation": aggregation,
        "group_by": group_by,
        "data": chart_data.to_dict(orient="records"),
    }

    return "```chart\n" + json.dumps(spec_dict, indent=2) + "\n```"
# ADD THIS FUNCTION after _run_growth_rate_aggregation() 
# (around line 1400-1500, before _build_chart_block())

# ============================================================================
# PROJECTION VS ACTUAL AGGREGATION FUNCTION
# ============================================================================
def _run_projection_vs_actual_aggregation(
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    mapping: Dict[str, Optional[str]],
    total_col: str,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    PROJECTION_VS_ACTUAL aggregation (correct timeline behavior).

    Key behaviors:
    - Actuals are summed across order rows.
    - Projections are de-duplicated at their natural grain (so they don't multiply by #orders).
    - Timeline is built from PROJECTIONS (or all weeks if asked), then actuals are outer-joined.
      This ensures weeks with Projected > 0 and Actual = 0 appear in the output.
    """

    group_by = spec.get("group_by") or []

    actual_col = mapping.get("total_mci")
    proj_col = mapping.get("proj_mci")
    week_col = mapping.get("week")  # often "Week number for Activity vs Projection"

    if not actual_col or actual_col not in base_df.columns:
        return None, float("nan")

    if not proj_col or proj_col not in base_df.columns:
        # If projections are truly not in df, return actuals only
        if not group_by:
            return None, float(base_df[actual_col].sum())
        group_cols = [mapping.get(g) for g in group_by if mapping.get(g)]
        if not group_cols and week_col and week_col in base_df.columns:
            group_cols = [week_col]
        if not group_cols:
            return None, float(base_df[actual_col].sum())
        grouped = base_df.groupby(group_cols, as_index=False)[actual_col].sum()
        grouped = grouped.rename(columns={actual_col: "Actual"})
        return grouped, float(base_df[actual_col].sum())

    df = base_df.copy()
    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce").fillna(0.0)
    # Keep projections as NaN until timeline merge; do NOT fill yet
    df[proj_col] = pd.to_numeric(df[proj_col], errors="coerce")

    # ------------------------------------------------------------
    # Determine grouping columns (default to Week)
    # ------------------------------------------------------------
    group_cols = [mapping.get(g) for g in group_by if mapping.get(g)]
    if not group_cols:
        if week_col and week_col in df.columns:
            group_cols = [week_col]
        else:
            # Scalar fallback
            actual_total = float(df[actual_col].sum())
            proj_total = float(df[proj_col].dropna().sum())
            variance = actual_total - proj_total
            variance_pct = (variance / proj_total * 100.0) if proj_total else float("nan")
            summary = pd.DataFrame([{
                "Actual": int(round(actual_total)),
                "Projected": int(round(proj_total)),
                "Variance": int(round(variance)),
                "Variance_%": round(variance_pct, 1),
            }])
            return summary, actual_total

    # ------------------------------------------------------------
    # 1) ACTUALS: sum across all order rows by group (e.g., week)
    # ------------------------------------------------------------
    actuals = (
        df.groupby(group_cols, as_index=False)[actual_col]
          .sum()
          .rename(columns={actual_col: "Actual"})
    )

    # ------------------------------------------------------------
    # 2) PROJECTIONS: de-duplicate before aggregating
    #    Because projection values were merged onto order rows and would multiply.
    # ------------------------------------------------------------
    # Build projection grain keys:
    # must include the group dimension (Week), plus identifying dims if present
    dedupe_keys = list(group_cols)
    for c in ["Year", "Distributor", "Catalogue description (sold as)"]:
        if c in df.columns and c not in dedupe_keys:
            dedupe_keys.append(c)

    proj_unique = (
        df[dedupe_keys + [proj_col]]
        .dropna(subset=[proj_col])               # only rows where projections matched
        .drop_duplicates(subset=dedupe_keys)     # critical to prevent multiplication
    )

    # If no matched projection rows, projections truly didn't join for this filtered slice
    # We will still return a timeline based on actuals only (unless you later add full-week calendar).
    if proj_unique.empty:
        proj_grouped = actuals[group_cols].copy()
        proj_grouped["Projected"] = 0.0
    else:
        proj_grouped = (
            proj_unique.groupby(group_cols, as_index=False)[proj_col]
                       .sum()
                       .rename(columns={proj_col: "Projected"})
        )

    # ------------------------------------------------------------
    # 3) TIMELINE: build from projections, then outer-join actuals
    #    This is the core fix for your ACCESOFARM “missing weeks” issue.
    # ------------------------------------------------------------
    timeline = proj_grouped.merge(actuals, on=group_cols, how="outer")

    # Fill missing values
    timeline["Projected"] = pd.to_numeric(timeline["Projected"], errors="coerce").fillna(0.0)
    timeline["Actual"] = pd.to_numeric(timeline["Actual"], errors="coerce").fillna(0.0)

    # ------------------------------------------------------------
    # 4) Variance calculations
    # ------------------------------------------------------------
    timeline["Variance"] = timeline["Actual"] - timeline["Projected"]
    timeline["Variance_%"] = timeline.apply(
        lambda row: (
            (row["Variance"] / row["Projected"] * 100.0) if row["Projected"] != 0
            else (float("nan") if row["Actual"] == 0 else float("inf"))
        ),
        axis=1
    ).round(1)

    # Format as ints for display
    for col in ["Actual", "Projected", "Variance"]:
        timeline[col] = timeline[col].round(0).astype(int)

    # Sort by group column (usually week)
    timeline = timeline.sort_values(group_cols)

    total_actual = float(timeline["Actual"].sum())
    return timeline, total_actual

def _build_core_answer_projection_vs_actual(
    group_df: Optional[pd.DataFrame],
    numeric_value: float,
    status_text: str,
    filter_text: str,
    aggregation: str,
) -> str:
    """
    Build the textual answer for projection_vs_actual aggregation.
    """
    if group_df is None:
        return (
            f"Based on {status_text} for {filter_text}, "
            f"the actual ordered amount is **{numeric_value:,.0f} mCi**."
        )
    
    preview_md = group_df.to_markdown(index=False)
    
    # Extract summary totals
    if "Actual" in group_df.columns:
        total_actual = int(group_df["Actual"].sum())
        total_projected = int(group_df["Projected"].sum()) if "Projected" in group_df.columns else 0
        total_variance = total_actual - total_projected
        variance_pct = (total_variance / total_projected * 100.0) if total_projected != 0 else 0.0
        
        header = (
            f"Here is the **weekly actual vs projected** for {filter_text} "
            f"({status_text}):\n\n"
            f"**Summary:**\n"
            f"- **Total Actual:** {total_actual:,} mCi\n"
            f"- **Total Projected:** {total_projected:,} mCi\n"
            f"- **Total Variance:** {total_variance:+,} mCi ({variance_pct:+.1f}%)\n\n"
            f"**Weekly Breakdown:**\n\n"
        )
    else:
        header = (
            f"Here is the **actual vs projected breakdown** for {filter_text} "
            f"({status_text}):\n\n"
        )
    
    return header + (preview_md or "No data available")

# --------------------------------------------------------------------
# 3) PUBLIC ENTRYPOINT – called from app.py
# --------------------------------------------------------------------
def _normalize_entity_filters(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a customer value actually exists in the distributor column,
    move it to distributor. Also checks the reverse.
    """
    filters = spec.get("filters") or {}

    # Build set of distributor names (normalized)
    if "Distributor" in df.columns:
        dist_names = {
            str(x).strip().lower()
            for x in df["Distributor"].dropna().unique()
        }
    else:
        dist_names = set()

    # Build set of customer names (normalized)
    if "Customer" in df.columns:
        cust_names = {
            str(x).strip().lower()
            for x in df["Customer"].dropna().unique()
        }
    else:
        cust_names = set()

    # Check if customer is actually a distributor
    cust_val = filters.get("customer")
    if cust_val:
        cust_lower = str(cust_val).strip().lower()
        if cust_lower in dist_names:
            print(f"🔧 NORMALIZING: Moving '{cust_val}' from customer → distributor")
            filters["distributor"] = cust_val
            filters["customer"] = None

    # Check if distributor is actually a customer
    dist_val = filters.get("distributor")
    if dist_val:
        dist_lower = str(dist_val).strip().lower()
        if dist_lower in cust_names and dist_lower not in dist_names:
            print(f"🔧 NORMALIZING: Moving '{dist_val}' from distributor → customer")
            filters["customer"] = dist_val
            filters["distributor"] = None

    spec["filters"] = filters
    return spec


def _disambiguate_customer_vs_distributor(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a best-effort distinction between 'customer' and 'distributor'.

    Rules:
      - If question mentions 'distributor', prefer distributor filter.
      - If question mentions 'customer' (and not 'distributor'), prefer customer filter.
      - If a 'customer' value matches or nearly matches a known distributor name,
        treat it as a distributor (especially important because projections live
        at distributor level).
    """
    filters = spec.get("filters") or {}
    question = (spec.get("_question_text") or "").lower()
    aggregation = spec.get("aggregation")

    customer_val = filters.get("customer")
    distributor_val = filters.get("distributor")

    # 1) Explicit language in question
    mentions_distributor = "distributor" in question or "distributors" in question
    mentions_customer = "customer" in question or "customers" in question

    if mentions_distributor and not mentions_customer:
        # Drop accidental customer
        if distributor_val:
            filters.pop("customer", None)
        spec["filters"] = filters
        return spec

    if mentions_customer and not mentions_distributor:
        # Drop accidental distributor
        if customer_val:
            filters.pop("distributor", None)
        spec["filters"] = filters
        return spec

    # 2) Build distributor & customer name sets for normalization
    dist_values: List[str] = []
    if "Distributor" in df.columns:
        dist_values = [str(x).strip() for x in df["Distributor"].dropna().unique()]
        distributor_names = {d.lower() for d in dist_values}
    else:
        distributor_names = set()

    cust_values: List[str] = []
    if "Customer" in df.columns:
        cust_values = [str(x).strip() for x in df["Customer"].dropna().unique()]
        customer_names = {c.lower() for c in cust_values}
    else:
        customer_names = set()

    # Helper for fuzzy token-based matching against distributors
    def _best_distributor_match(name: str) -> Optional[str]:
        name_norm = str(name).strip().lower()
        name_tokens = {t for t in name_norm.replace("-", " ").split() if len(t) > 1}
        if not name_tokens or not dist_values:
            return None

        best_match = None
        best_score = 0

        for dist in dist_values:
            dist_tokens = {t for t in dist.lower().replace("-", " ").split() if len(t) > 1}
            overlap = name_tokens & dist_tokens
            score = len(overlap)
            if score > best_score:
                best_score = score
                best_match = dist

        return best_match if best_score > 0 else None

    # 3) Customer → Distributor promotion (this is what fixes PI Medical)
    if customer_val and not distributor_val:
        cust_norm = str(customer_val).strip().lower()

        # 3a) Exact match against distributor names
        if cust_norm in distributor_names:
            canonical = next((d for d in dist_values if d.lower() == cust_norm), customer_val)
            print(
                f"🔧 NORMALIZING: Moving '{customer_val}' from customer → distributor "
                f"(exact match: '{canonical}')"
            )
            filters.pop("customer", None)
            filters["distributor"] = canonical
            spec["filters"] = filters
            return spec

        # 3b) Fuzzy match against distributor names (token overlap)
        #     This is especially important for projection_vs_actual since projections are
        #     defined at distributor level.
        best_match = _best_distributor_match(customer_val)
        if best_match:
            print(
                f"🔧 NORMALIZING: Treating customer '{customer_val}' as distributor '{best_match}' "
                f"(token overlap)"
            )
            filters.pop("customer", None)
            filters["distributor"] = best_match
            spec["filters"] = filters
            return spec

    # 4) Distributor → Customer fallback (rare, but keep the original safety)
    if distributor_val:
        dist_norm = str(distributor_val).strip().lower()
        if dist_norm in customer_names and dist_norm not in distributor_names:
            print(
                f"🔧 NORMALIZING: Moving '{distributor_val}' from distributor → customer "
                "(found only in Customer column)"
            )
            filters["customer"] = distributor_val
            filters["distributor"] = None

    spec["filters"] = filters
    return spec


def _normalize_product_filters(spec: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Prevents non-product values (like distributor names) from being
    misclassified as product filters.
    """
    filters = spec.get("filters") or {}

    product_val = filters.get("product_sold") or filters.get("product_catalogue")

    if product_val:
        pv = str(product_val).lower()

        # Very rough markers that indicate a REAL product, not a company name
        isotope_markers = [
            "lu", "lutetium", "177", "tb", "terbium",
            "psma", "chloride", "n.c.a", "c.a", "nca"  # ← ADDED "nca" HERE
        ]

        is_real_product = any(k in pv for k in isotope_markers)

        if not is_real_product:
            # DSD, PI Medical, etc → drop as product filters
            print(f"🧹 Dropping invalid product filter: '{product_val}' (not a real product)")
            filters.pop("product_sold", None)
            filters.pop("product_catalogue", None)

    spec["filters"] = filters
    return spec

def _reshape_for_display(group_df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Reshape aggregated data into a more pivot-like layout when possible:
    - Put time dimensions (Year, Week, Quarter, Half year, Month) in columns.
    - Keep non-time dimensions (Distributor, Region, etc.) as rows.
    Works for sum, average and grouped growth_rate outputs.
    """
    if group_df is None or group_df.empty:
        return group_df

    # ✅ NEW: Don't pivot growth rate tables - they need row format
    aggregation = spec.get("aggregation", "sum_mci")
    if aggregation == "growth_rate":
        return group_df  # Return as-is for growth calculations

    df = group_df.copy()

    # Identify time dimensions
    time_cols_priority = ["Year", "Quarter", "Month", "Week", "Half year"]
    time_cols = [c for c in time_cols_priority if c in df.columns]

    if not time_cols:
        # Nothing to pivot on
        return df

    # Use a single dominant time dimension (Year > Quarter > Month > Week > Half year)
    time_col = time_cols[0]

    # Identify metric columns (mCi, growth, abs change, averages)
    metric_candidates = [
        c
        for c in df.columns
        if any(
            k in str(c).lower()
            for k in ["mci", "growthrate", "abschange", "average"]
        )
    ]

    if not metric_candidates:
        return df

    # Non-time, non-metric dimensions become the row index
    non_time_non_metric = [
        c for c in df.columns if c not in [time_col] + metric_candidates
    ]
    idx_cols = non_time_non_metric or None

    if len(metric_candidates) == 1:
        metric = metric_candidates[0]
        if idx_cols:
            # Example: Distributor x Year → Distributor row, years as columns
            pivot_df = df.pivot(
                index=idx_cols,
                columns=time_col,
                values=metric,
            ).reset_index()
        else:
            # Example: NCA per Year → single row, years as columns
            pivot_df = df.pivot_table(
                index=None,
                columns=time_col,
                values=metric,
                aggfunc="sum",
            ).reset_index(drop=True)
    else:
        # Multiple metrics: keep as-is for now (like your weekly growth table)
        return df

    pivot_df.columns = [str(c) for c in pivot_df.columns]
    return pivot_df
    
def _refine_answer_text(
    client,
    raw_answer: str,
    question: str,
) -> str:
    """
    Optional stylistic refinement step:
    - Preserve ALL numbers.
    - Preserve ALL markdown tables.
    - Preserve any ```chart``` blocks.
    - Only improve wording, clarity, and flow.
    """
    if client is None:
        return raw_answer

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # or whatever small/cheap model you're using
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are polishing an analytics answer for a radiopharmaceuticals dashboard.\n"
                        "REQUIREMENTS:\n"
                        "- Do NOT change any numbers.\n"
                        "- Do NOT change any markdown tables (rows, columns, or values).\n"
                        "- Do NOT change any ```chart code blocks.\n"
                        "- Do NOT add new analysis or interpretations.\n"
                        "You may only improve wording, tone, and readability of the surrounding text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{question}\n\n"
                        f"Current answer in markdown:\n{raw_answer}"
                    ),
                },
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else raw_answer
    except Exception:
        # Fail-safe: if refinement fails for any reason, fall back to raw answer
        return raw_answer

def _normalize_entity_filters(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a customer value actually exists in the distributor column,
    move it to distributor. This fixes cases where the LLM or injection
    logic misclassified a distributor as a customer.
    
    Also checks the reverse: if a distributor value is actually a customer.
    """
    filters = spec.get("filters") or {}
    
    # Build set of distributor names (normalized)
    if "Distributor" in df.columns:
        dist_names = {
            str(x).strip().lower() 
            for x in df["Distributor"].dropna().unique()
        }
    else:
        dist_names = set()
    
    # Build set of customer names (normalized)
    if "Customer" in df.columns:
        cust_names = {
            str(x).strip().lower() 
            for x in df["Customer"].dropna().unique()
        }
    else:
        cust_names = set()
    
    # Check if customer is actually a distributor
    cust_val = filters.get("customer")
    if cust_val:
        cust_lower = str(cust_val).strip().lower()
        if cust_lower in dist_names:
            print(f"[NORMALIZE] Moving '{cust_val}' from customer to distributor")
            filters["distributor"] = cust_val
            filters["customer"] = None
    
    # Check if distributor is actually a customer
    dist_val = filters.get("distributor")
    if dist_val:
        dist_lower = str(dist_val).strip().lower()
        if dist_lower in cust_names and dist_lower not in dist_names:
            print(f"[NORMALIZE] Moving '{dist_val}' from distributor to customer")
            filters["customer"] = dist_val
            filters["distributor"] = None
    
    spec["filters"] = filters
    return spec


def _inject_customer_from_question(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to infer entity filters from the question text.
    PRIORITY ORDER:
    1. If distributor filter is already set, skip customer injection
    2. Try to match as a DISTRIBUTOR first (check Distributor column) - AGGRESSIVE matching
    3. Fall back to matching as a CUSTOMER (check Customer column)
    """
    filters = spec.get("filters") or {}
    question = (spec.get("_question_text") or "").lower()
    
    print(f"\n[INJECT] Starting entity injection")
    print(f"  Question: {question}")
    
    if "Distributor" in df.columns:
        unique_distributors = [str(x).strip() for x in df["Distributor"].dropna().unique()]
        print(f"  Found {len(unique_distributors)} unique distributors")
    
    if "Customer" in df.columns:
        unique_customers = [str(x).strip() for x in df["Customer"].dropna().unique()]
        print(f"  Found {len(unique_customers)} unique customers")
    
    # If distributor is already set, don't try to find a customer
    if filters.get("distributor"):
        print(f"  Distributor already set: {filters['distributor']}, skipping")
        return spec
    
    # If customer is already set, don't override it
    if filters.get("customer"):
        print(f"  Customer already set: {filters['customer']}, skipping")
        return spec

    # ===== STEP 1: Try to match as DISTRIBUTOR (VERY AGGRESSIVE) =====
    if "Distributor" in df.columns:
        unique_distributors = [str(x).strip() for x in df["Distributor"].dropna().unique()]
        
        # Tokenize question into keywords
        q_tokens = set(
            t.strip().lower() 
            for t in question.replace(",", " ").replace("'", "").split() 
            if len(t.strip()) > 2
        )
        
        # Try exact match first (case-insensitive)
        for dist in unique_distributors:
            dist_lower = dist.lower()
            if dist_lower in question:
                print(f"  [FOUND] Distributor (exact): {dist}")
                filters["distributor"] = dist
                spec["filters"] = filters
                return spec
        
        # Try token-based matching (VERY LOOSE - any token overlap)
        best_match = None
        best_score = 0

        for dist in unique_distributors:
            dist_lower = dist.lower()
            dist_tokens = set(dist_lower.split())
            
            # Count how many dist tokens appear in question tokens
            overlap = dist_tokens & q_tokens
            score = len(overlap)
            
            if score > best_score:
                best_score = score
                best_match = dist

        if best_match and best_score > 0:
            print(f"  [FOUND] Distributor (token match): {best_match}")
            filters["distributor"] = best_match
            spec["filters"] = filters
            return spec

    # ===== STEP 2: Fall back to matching as CUSTOMER =====
    if "Customer" not in df.columns:
        print(f"  No Customer column in df")
        return spec

    unique_customers = [str(x).strip() for x in df["Customer"].dropna().unique()]
    
    # Try exact match first (case-insensitive)
    for cust in unique_customers:
        cust_lower = cust.lower()
        if cust_lower in question:
            print(f"  [FOUND] Customer (exact): {cust}")
            filters["customer"] = cust
            spec["filters"] = filters
            return spec
    
    # Try partial match
    q_tokens = [t.strip() for t in question.replace(",", " ").split() if len(t.strip()) > 2]
    best_match = None
    best_score = 0

    for cust in unique_customers:
        cust_lower = cust.lower()
        score = 0
        for tok in q_tokens:
            if tok in cust_lower or cust_lower in tok:
                score += 1
        if score > best_score:
            best_score = score
            best_match = cust

    if best_match and best_score > 0:
        print(f"  [FOUND] Customer (partial): {best_match}")
        filters["customer"] = best_match
        spec["filters"] = filters
        return spec
    
    print(f"  [NOT FOUND] No entity match")
    return spec
# Add these TWO functions right BEFORE answer_question_from_df()
# (around line 1750-1800, before the big answer_question_from_df function)

def _pivot_growth_by_entity(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Pivot growth rate results so that:
    - Rows = time dimension (Week, Year, Month, etc.)
    - Columns = entities (Distributor, Product, etc.) with their growth rates
    
    This makes it much easier to compare growth across entities.
    """
    if df is None or df.empty:
        print(f"[PIVOT] Input df is None or empty")
        return df
    
    compare = spec.get("compare") or {}
    entity_type = compare.get("entity_type")
    aggregation = spec.get("aggregation")
    
    print(f"[PIVOT] Input shape: {df.shape}, columns: {list(df.columns)}")
    print(f"[PIVOT] entity_type: {entity_type}, aggregation: {aggregation}")
    
    # Only pivot if:
    # 1. This is a growth_rate aggregation
    # 2. We're comparing entities
    # 3. We have a growth column
    if aggregation != "growth_rate" or not entity_type:
        print(f"[PIVOT] Skipping - not growth_rate or no entity_type")
        return df
    
    growth_col = None
    time_col = None
    
    if "WoW_Growth_%" in df.columns:
        growth_col = "WoW_Growth_%"
        time_col = "Week"
        print(f"[PIVOT] Detected WoW_Growth_% - using Week as time_col")
    elif any("Growth" in col and "%" in col for col in df.columns):
        growth_col = [c for c in df.columns if "Growth" in c and "%" in c][0]
        time_col = "Year" if "Year" in df.columns else None
        print(f"[PIVOT] Detected {growth_col} - using {time_col} as time_col")
    
    if growth_col is None or time_col not in df.columns:
        print(f"[PIVOT] Skipping - no growth_col or time_col not in df")
        print(f"[PIVOT]   growth_col={growth_col}, time_col={time_col}")
        print(f"[PIVOT]   df.columns={list(df.columns)}")
        return df
    
    df = df.copy()
    
    # Find the entity column in the dataframe
    entity_col = None
    if entity_type == "distributor":
        entity_col = "Distributor"
    elif entity_type == "product_sold":
        entity_col = "Catalogue description (sold as)"
    elif entity_type == "product_catalogue":
        entity_col = "Catalogue description"
    
    print(f"[PIVOT] Looking for entity_col: {entity_col}")
    
    # Fallback: search for it
    if entity_col not in df.columns:
        print(f"[PIVOT] entity_col '{entity_col}' not in df.columns, searching...")
        possible_cols = [c for c in df.columns if entity_type.replace("_", " ").lower() in c.lower()]
        if possible_cols:
            entity_col = possible_cols[0]
            print(f"[PIVOT] Found possible column: {entity_col}")
        else:
            print(f"[PIVOT] No possible columns found, returning unpivoted")
            return df
    
    if entity_col not in df.columns:
        print(f"[PIVOT] entity_col '{entity_col}' still not in df, giving up")
        print(f"[PIVOT] Available columns: {list(df.columns)}")
        return df
    
    print(f"[PIVOT] Using entity_col: {entity_col}")
    print(f"[PIVOT] Using growth_col: {growth_col}")
    print(f"[PIVOT] Using time_col: {time_col}")
    print(f"[PIVOT] Unique entities before pivot: {df[entity_col].unique()}")
    
    # Simplify names for display
    name_simplifications = {
        "dsd pharma gmbh": "DSD",
        "dsd": "DSD",
        "pi medical diagnostic equipment b.v.": "PI Medical",
        "pi medical": "PI Medical",
    }
    
    df[entity_col] = (
        df[entity_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: name_simplifications.get(x, x.title()))
    )
    
    print(f"[PIVOT] After simplification: {df[entity_col].unique()}")
    
    # Pivot: rows = time_col, columns = entity, values = growth_col
    try:
        print(f"[PIVOT] Calling pivot_table with index={time_col}, columns={entity_col}, values={growth_col}")
        pivot_df = df.pivot_table(
            index=time_col,
            columns=entity_col,
            values=growth_col,
            aggfunc="first"  # If somehow multiple rows, take first
        )
        print(f"[PIVOT] Pivot successful! Shape before reset: {pivot_df.shape}")
        
        pivot_df = pivot_df.reset_index()
        pivot_df.columns.name = None  # Remove the category name from column header
        
        print(f"[PIVOT] After reset_index: shape={pivot_df.shape}, columns={list(pivot_df.columns)}")
        
        # Round growth percentages to 1 decimal
        for col in pivot_df.columns:
            if col != time_col:
                pivot_df[col] = pd.to_numeric(pivot_df[col], errors="coerce").round(1)
        
        print(f"[PIVOT] ✅ Successfully pivoted! Final shape: {pivot_df.shape}")
        return pivot_df
    except Exception as e:
        print(f"[PIVOT] ❌ Could not pivot growth table: {e}")
        import traceback
        traceback.print_exc()
        return df

# ============================================
# MAIN ENTRYPOINT
# ============================================
def answer_question_from_df(
    question: str,
    consolidated_df: pd.DataFrame,
    history: Optional[List[Dict[str, str]]] = None,
    client=None,
) -> str:
    """Main function to interpret, execute, and answer a question from the DataFrame."""
    if consolidated_df is None or consolidated_df.empty:
        return "The consolidated data is empty. Please load data first."

    print("\n" + "=" * 70)
    print("DEBUG answer_question_from_df() START")
    print("=" * 70)
    print(f"Question: {question}")

    # 1) Build & normalize the spec
    spec = _interpret_question_with_llm(question, history=history) or {}

    # Defensive normalization so we never get None where dicts are expected
    spec.setdefault("filters", {})
    spec.setdefault("time_window", {})
    spec.setdefault("compare", {})

    print(f"\n[SPEC] After interpretation:")
    print(f"  aggregation: {spec.get('aggregation')}")
    print(f"  group_by: {spec.get('group_by')}")
    print(f"  filters.customer: {spec.get('filters', {}).get('customer')}")
    print(f"  filters.distributor: {spec.get('filters', {}).get('distributor')}")
    print(f"  filters.year: {spec.get('filters', {}).get('year')}")

    # ------------------------------------------------------------------
    # 1a) Extract explicit week / year from the question text if missing
    # ------------------------------------------------------------------
    q_lower = (question or "").lower()
    filters = spec.get("filters") or {}

    # If LLM didn't set week, try to extract "week 35", "week 7", etc.
    if not filters.get("week"):
        m_week = re.search(r"\bweek\s+(\d{1,2})\b", q_lower)
        if m_week:
            filters["week"] = int(m_week.group(1))
            print(f"  [EXTRACT] Week from text: {filters['week']}")

    # If LLM didn't set year, try to extract 2024, 2025, etc.
    if not filters.get("year"):
        m_year = re.search(r"\b(20[2-3][0-9])\b", q_lower)
        if m_year:
            filters["year"] = int(m_year.group(1))
            print(f"  [EXTRACT] Year from text: {filters['year']}")

    spec["filters"] = filters

    # Ensure we have a year for "why" questions with an explicit week but no year
    if spec.get("_why_question") and filters.get("week") and not filters.get("year"):
        if "Year" in consolidated_df.columns and not consolidated_df["Year"].dropna().empty:
            filters["year"] = int(consolidated_df["Year"].max())

    # ------------------------------------------------------------------
    # 1b) Shipping-status and entity normalization
    # ------------------------------------------------------------------
    spec = _force_cancellation_status_from_text(question, spec)
    spec = _ensure_all_statuses_when_grouped(spec)

    print(f"\n[ENTITY] Before entity processing:")
    print(f"  customer: {spec.get('filters', {}).get('customer')}")
    print(f"  distributor: {spec.get('filters', {}).get('distributor')}")

    spec = _inject_customer_from_question(consolidated_df, spec)
    spec = _normalize_entity_filters(consolidated_df, spec)
    spec = _disambiguate_customer_vs_distributor(consolidated_df, spec)
    spec = _normalize_product_filters(spec, question)

    print(f"\n[ENTITY] After entity processing:")
    print(f"  customer: {spec.get('filters', {}).get('customer')}")
    print(f"  distributor: {spec.get('filters', {}).get('distributor')}")

    # Final fix for "why / reason / drop" style questions
    if spec.get("_why_question"):
        if spec.get("shipping_status_mode") in (None, "all"):
            spec["shipping_status_mode"] = "countable"
            spec["shipping_status_list"] = []

        gb = spec.get("group_by") or []
        if gb == ["shipping_status"]:
            gb = []
        elif "shipping_status" in gb and len(gb) > 1:
            gb = [g for g in gb if g != "shipping_status"]
        spec["group_by"] = gb

    # ------------------------------------------------------------------
    # 2) Apply filters (with special handling for growth_rate & compare)
    # ------------------------------------------------------------------
    spec_for_filtering = dict(spec)  # shallow copy is fine here

    # CRITICAL FIX: Clear distributor filter when comparing distributors
    if spec.get("aggregation") == "compare":
        compare = spec.get("compare") or {}
        if compare.get("entity_type") == "distributor":
            print(f"\n[FIX] Clearing distributor filter for comparison")
            print(f"  Entities: {compare.get('entities')}")
            
            filters = spec_for_filtering.get("filters") or {}
            old_dist = filters.get("distributor")
            filters["distributor"] = None
            spec_for_filtering["filters"] = filters
            
            print(f"  Changed distributor filter from '{old_dist}' → None")
            print(f"  This allows BOTH distributors' data to be included")

    # Growth rate: clear year filter to include all years
    if spec.get("aggregation") == "growth_rate":
        group_by = spec.get("group_by") or []
        if "year" in group_by:
            filters_f = spec_for_filtering.get("filters") or {}
            year_filter = filters_f.get("year")
            if year_filter:
                print(f"[YoY] Clearing year filter to include all years")
                filters_f["year"] = None
                spec_for_filtering["filters"] = filters_f

    df_filtered = _apply_filters(consolidated_df, spec_for_filtering)

    print(f"\n[FILTER] After filtering:")
    print(f"  rows returned: {len(df_filtered) if df_filtered is not None else 0}")

    # ------------------------------------------------------------------
    # 3) Run aggregation
    # ------------------------------------------------------------------
    group_df, numeric_value = _run_aggregation(df_filtered, spec, consolidated_df)

    print(f"\n[AGGREGATION] After aggregation:")
    if group_df is not None:
        print(f"  result shape: {group_df.shape}")
        print(f"  columns: {list(group_df.columns)}")
    else:
        print(f"  result: None")
    print(f"  numeric value: {numeric_value}")

    # Define aggregation early for safe use throughout
    aggregation = spec.get("aggregation", "sum_mci") or "sum_mci"
    print(f"\n[DEBUG] aggregation set to: {aggregation}")

    # ------------------------------------------------------------------
    # 4) Reshape / pivot for display
    # ------------------------------------------------------------------
    # 4a) Growth-rate tables: pivot entities vs periods first
    if group_df is not None and aggregation == "growth_rate":
        print(f"\n[PIVOT] Attempting to pivot growth rate table...")
        original_shape = group_df.shape
        group_df = _pivot_growth_by_entity(group_df, spec)
        print(
            f"[PIVOT] After pivot: shape changed from {original_shape} "
            f"to {group_df.shape if group_df is not None else None}"
        )
    else:
        # 4b) Regular reshaping (week / month / year on separate columns)
        group_df = _reshape_for_display(group_df, spec)

    # 4c) First numeric formatting pass
    if group_df is not None and not group_df.empty:
        for col in group_df.columns:
            col_lower = str(col).lower()
            is_year_col = False
            try:
                is_year_col = str(int(col)) == str(col)
            except Exception:
                pass

            if (
                "mci" in col_lower
                or "growthrate" in col_lower
                or "abschange" in col_lower
                or is_year_col
            ):
                try:
                    group_df[col] = (
                        pd.to_numeric(group_df[col], errors="coerce")
                        .fillna(0)
                        .round(0)
                        .astype(int)
                    )
                except Exception:
                    pass

    # 4d) If we have Year + Total_mCi, pivot so years are columns
    group_by = spec.get("group_by") or []
    if (
        group_df is not None
        and not group_df.empty
        and "Year" in group_df.columns
        and "Total_mCi" in group_df.columns
    ):
        non_year_cols = [c for c in group_df.columns if c not in ("Year", "Total_mCi")]

        if non_year_cols:
            try:
                pivot_df = group_df.pivot(
                    index=non_year_cols,
                    columns="Year",
                    values="Total_mCi",
                ).reset_index()
                pivot_df.columns = [str(c) for c in pivot_df.columns]
                group_df = pivot_df
            except Exception:
                pass
        else:
            years = sorted(group_df["Year"].unique())
            wide = pd.DataFrame(
                [
                    {
                        str(y): float(
                            group_df.loc[group_df["Year"] == y, "Total_mCi"].sum()
                        )
                        for y in years
                    }
                ]
            )
            group_df = wide

    # 4e) Second numeric formatting pass (after possible year pivot)
    if group_df is not None and not group_df.empty:
        for col in group_df.columns:
            col_lower = str(col).lower()
            is_year_col = False
            try:
                is_year_col = str(int(col)) == str(col)
            except Exception:
                pass

            if (
                "mci" in col_lower
                or "growthrate" in col_lower
                or "abschange" in col_lower
                or is_year_col
            ):
                try:
                    group_df[col] = (
                        pd.to_numeric(group_df[col], errors="coerce")
                        .fillna(0)
                        .round(0)
                        .astype(int)
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 5) Human-readable filter + status text
    # ------------------------------------------------------------------
    filters = spec.get("filters", {}) or {}

    parts = []
    if filters.get("customer"):
        parts.append(f"customer **{filters['customer']}**")
    if filters.get("distributor"):
        parts.append(f"distributor **{filters['distributor']}**")
    if filters.get("country"):
        parts.append(f"country **{filters['country']}**")
    if filters.get("region"):
        parts.append(f"region **{filters['region']}**")
    if filters.get("year"):
        parts.append(f"year **{filters['year']}**")
    if filters.get("month"):
        parts.append(f"month **{filters['month']}**")
    if filters.get("quarter"):
        parts.append(f"quarter **{filters['quarter']}**")
    if filters.get("half_year"):
        parts.append(f"half-year **{filters['half_year']}**")
    if filters.get("product_sold"):
        parts.append(f"product sold as **{filters['product_sold']}**")
    if filters.get("product_catalogue"):
        parts.append(f"product from catalogue **{filters['product_catalogue']}**")
    if filters.get("production_site"):
        parts.append(f"production site **{filters['production_site']}**")

    filter_text = ", ".join(parts) if parts else "all records"

    ship_mode = spec.get("shipping_status_mode", "countable")
    if ship_mode == "countable":
        status_text = (
            "countable orders (shipped, partially shipped, shipped late, or being processed)"
        )
    elif ship_mode == "cancelled":
        status_text = "cancelled or rejected orders"
    elif ship_mode == "explicit":
        status_text = "orders with the specified shipping statuses"
    else:
        status_text = f"orders with '{ship_mode}' statuses"

    row_count = len(df_filtered) if df_filtered is not None else 0

# ------------------------------------------------------------------
    # 6) Build the core textual answer by aggregation type
    # ------------------------------------------------------------------
    core_answer = ""
# In answer_question_from_df(), find the COMPARE section and replace it with:

    # COMPARE - distributor/product comparison
# PROJECTION VS ACTUAL
    if aggregation == "projection_vs_actual":
        if group_df is None:
            core_answer = (
                f"Based on {status_text} for {filter_text}, "
                f"the actual ordered amount is **{numeric_value:,.0f} mCi**."
            )
        else:
            # Check if this is an "actual only" result (no projections)
            has_projection = "Projected" in group_df.columns and group_df["Projected"].sum() > 0
            
            if not has_projection:
                # Only actuals, no projections available
                preview_md = group_df.to_markdown(index=False)
                core_answer = (
                    f"Projections are not available for {filter_text}. "
                    f"Here are the **actual orders** for {status_text}:\n\n"
                    + (preview_md or "No data available")
                )
            else:
                # Full projection vs actual comparison
                preview_md = group_df.to_markdown(index=False)
                
                # Extract summary totals
                if "Actual" in group_df.columns:
                    total_actual = int(group_df["Actual"].sum())
                    total_projected = int(group_df["Projected"].sum()) if "Projected" in group_df.columns else 0
                    total_variance = total_actual - total_projected
                    variance_pct = (total_variance / total_projected * 100.0) if total_projected != 0 else 0.0
                    
                    header = (
                        f"Here is the **weekly actual vs projected** for {filter_text} "
                        f"({status_text}):\n\n"
                        f"**Summary:**\n"
                        f"- **Total Actual:** {total_actual:,} mCi\n"
                        f"- **Total Projected:** {total_projected:,} mCi\n"
                        f"- **Total Variance:** {total_variance:+,} mCi ({variance_pct:+.1f}%)\n\n"
                        f"**Weekly Breakdown:**\n\n"
                    )
                    core_answer = header + (preview_md or "No data available")
                else:
                    core_answer = (
                        f"Here is the **actual vs projected breakdown** for {filter_text} "
                        f"({status_text}):\n\n"
                        + (preview_md or "No data available")
                    )

    # COMPARE - distributor/product comparison
    elif aggregation == "compare":
        if group_df is None:
            core_answer = (
                f"Based on {status_text} for {filter_text}, "
                f"the total ordered amount is **{numeric_value:,.0f} mCi**."
            )
        else:
            preview_md = group_df.to_markdown(index=False)
            compare = spec.get("compare") or {}
            entity_type = compare.get("entity_type", "entities")
            entities = compare.get("entities", [])
            
            # Humanize the entity type
            if entity_type == "distributor":
                entity_label = "distributors"
            elif entity_type == "product_sold":
                entity_label = "products (sold as)"
            elif entity_type == "product_catalogue":
                entity_label = "products (catalogue)"
            else:
                entity_label = entity_type
            
            # Build the header
            if entities and len(entities) > 0:
                entity_list = " vs ".join([f"**{e}**" for e in entities])
                header = (
                    f"Here is a **side-by-side comparison** of {entity_label} "
                    f"({entity_list}) for {status_text} for {filter_text}:\n\n"
                )
            else:
                header = (
                    f"Here is a **side-by-side comparison** for {status_text} for {filter_text}:\n\n"
                )
            
            core_answer = header + (preview_md or "No data available")

    # SUM (default), TOP N
    elif aggregation in ("sum_mci", "top_n"):
        if group_df is None:
            core_answer = (
                f"Based on {status_text} for {filter_text}, the total ordered amount is "
                f"**{numeric_value:,.0f} mCi**, calculated from **{row_count}** rows."
            )
        else:
            preview_md = group_df.to_markdown(index=False)
            header = (
                f"Here is a breakdown for {status_text} for {filter_text}. "
                f"The total across all groups is **{numeric_value:,.0f} mCi** "
                f"(from **{row_count}** rows).\n\n"
            )
            core_answer = header + preview_md

    # AVERAGE
    elif aggregation == "average_mci":
        if group_df is None:
            core_answer = (
                f"Based on {status_text} for {filter_text}, the **average ordered amount** per row is "
                f"**{numeric_value:,.0f} mCi**, across **{row_count}** rows."
            )
        else:
            preview_md = group_df.to_markdown(index=False)
            header = (
                f"Here is the **average ordered amount per group** for {status_text} for {filter_text}. "
                f"The overall average across all rows is "
                f"**{numeric_value:,.0f} mCi** (from **{row_count}** rows).\n\n"
            )
            core_answer = header + preview_md

    # SHARE OF TOTAL
    elif aggregation == "share_of_total":
        share_debug = spec.get("_share_debug", {}) or {}
        if group_df is None:
            share_pct = numeric_value * 100.0
            num = share_debug.get("numerator")
            den = share_debug.get("denominator")

            if num is not None and den is not None and den != 0:
                core_answer = (
                    f"Based on {status_text} for {filter_text}, the ordered amount is "
                    f"**{num:,.0f} mCi**, which represents **{share_pct:.1f}%** "
                    f"of the corresponding total (**{den:,.0f} mCi**)."
                )
            else:
                core_answer = (
                    f"Based on {status_text} for {filter_text}, the share of total is "
                    f"approximately **{share_pct:.1f}%**."
                )
        else:
            den = share_debug.get("denominator")
            if den is not None and den != 0:
                denom_txt = f"The total ordered amount (denominator) is **{den:,.0f} mCi**."
            else:
                denom_txt = "The total ordered amount (denominator) is derived from the current filters."
            preview_md = group_df.to_markdown(index=False)
            header = (
                f"Here is the **share of total ordered amount per group** for {status_text} "
                f"for {filter_text}. {denom_txt}\n\n"
            )
            core_answer = header + preview_md

    # GROWTH RATE
    elif aggregation == "growth_rate":
        if group_df is None:
            if numeric_value is None or pd.isna(numeric_value):
                core_answer = (
                    f"Could not compute a growth rate for {status_text} for {filter_text}."
                )
            else:
                pct = numeric_value * 100.0
                core_answer = (
                    f"Based on {status_text} for {filter_text}, the growth rate is "
                    f"**{pct:.1f}%**."
                )
        else:
            cols = list(group_df.columns)

            # Week-over-week growth table
            if "WoW_Growth_%" in cols:
                preview_md = group_df.to_markdown(index=False)
                time_window = spec.get("time_window") or {}
                n_weeks = time_window.get("n_weeks")

                if n_weeks:
                    header = (
                        f"Here is the **week-over-week growth** for the last {n_weeks} weeks "
                        f"for {filter_text}.\n\n"
                    )
                else:
                    header = (
                        f"Here is the **week-over-week growth** "
                        f"for {filter_text}.\n\n"
                    )

                core_answer = header + (preview_md or "")

            # Year-over-year growth table
            elif "YoY_Growth" in cols or any("vs" in str(c) and "Growth" in str(c) for c in cols):
                preview_md = group_df.to_markdown(index=False)
                header = (
                    f"Here is the **year-over-year growth** "
                    f"for {filter_text}.\n\n"
                )
                core_answer = header + (preview_md or "")

            else:
                preview_md = group_df.to_markdown(index=False)
                header = f"Here is the **growth breakdown** for {filter_text}:\n\n"
                core_answer = header + (preview_md or "")

    # Fallback
    else:
        if group_df is None:
            core_answer = (
                f"Based on {status_text} for {filter_text}, "
                f"the value is **{numeric_value:,.0f}** across **{row_count}** rows."
            )
        else:
            preview_md = group_df.to_markdown(index=False)
            core_answer = (
                f"Here is the breakdown for {status_text} for {filter_text}:\n\n"
                + preview_md
            )

    # ------------------------------------------------------------------
    # 7) Optional metadata + chart blocks
    # ------------------------------------------------------------------
    meta_block = None
    try:
        if _should_include_metadata(spec):
            meta_block = _build_metadata_snippet(df_filtered, spec)
    except NameError:
        meta_block = None

    chart_block = None
    try:
        if group_df is not None and not group_df.empty:
            chart_block = _build_chart_block(group_df, spec, aggregation)
    except NameError:
        chart_block = None

    final_answer = core_answer

    if meta_block:
        final_answer += meta_block

    if chart_block:
        final_answer += "\n\n" + chart_block

    # ------------------------------------------------------------------
    # 8) Optional refinement pass
    # ------------------------------------------------------------------
    try:
        refined_answer = _refine_answer_text(client, final_answer, question)
    except Exception:
        refined_answer = final_answer

    return refined_answer

# -------------------------
# Dynamic week window logic
# -------------------------
def _apply_time_window(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies dynamic week window filters based on the 'time_window' structure in spec.
    This handles:
    - last N weeks
    - anchored last N weeks

    Expected structure in spec:
    spec["time_window"] = {
        "mode": "last_n_weeks" | "anchored_last_n_weeks",
        "n_weeks": int,
        "anchor": {
            "year": int | None,
            "week": int | None
        }
    }
    
    Args:
        df: DataFrame to filter
        spec: The spec dictionary containing time_window configuration
    
    Returns:
        Filtered DataFrame with only rows matching the time window
    """
    if df is None or df.empty:
        return df
    
    tw = spec.get("time_window") or {}
    mode = tw.get("mode")
    n_weeks = tw.get("n_weeks")
    anchor = tw.get("anchor") or {}
    anchor_year = anchor.get("year")
    anchor_week = anchor.get("week")

    # If no dynamic mode: nothing to do
    if not mode or n_weeks is None:
        return df

    # Be defensive: n_weeks and anchor_week might arrive as strings
    try:
        n_weeks = int(n_weeks)
    except (ValueError, TypeError):
        return df  # invalid -> ignore gracefully

    df = df.copy()

    # Check that Year and Week columns exist
    if "Year" not in df.columns or "Week" not in df.columns:
        return df.iloc[0:0]  # Empty result if no time columns

    # Determine the effective year to operate in
    filters = spec.get("filters") or {}
    effective_year = filters.get("year")

    if effective_year is None and anchor_year is not None:
        try:
            effective_year = int(anchor_year)
        except (ValueError, TypeError):
            effective_year = None

    if effective_year is None:
        # Fall back to latest year present in the data
        if df["Year"].dropna().empty:
            return df.iloc[0:0]
        try:
            effective_year = int(df["Year"].max())
        except (ValueError, TypeError):
            return df.iloc[0:0]

    df_year = df[df["Year"] == effective_year]

    if df_year.empty:
        # No data for that year -> empty result
        return df.iloc[0:0]

    # Get available weeks for this year
    try:
        available_weeks = df_year["Week"].dropna().unique()
        if len(available_weeks) == 0:
            return df.iloc[0:0]
        min_week = int(df_year["Week"].min())
        max_week = int(df_year["Week"].max())
    except (ValueError, TypeError):
        return df.iloc[0:0]

    # Determine anchor week (end of the window)
    if mode == "anchored_last_n_weeks":
        if anchor_week is None:
            return df.iloc[0:0]
        try:
            end_week = int(anchor_week)
        except (ValueError, TypeError):
            return df.iloc[0:0]
    else:
        # last_n_weeks -> use the max week of that year
        end_week = max_week

    # Compute start week and clamp to minimum week present
    start_week = end_week - n_weeks + 1
    if start_week < min_week:
        start_week = min_week

    # Filter by year and week range
    df_filtered = df_year[
        (df_year["Week"] >= start_week) & (df_year["Week"] <= end_week)
    ]

    return df_filtered