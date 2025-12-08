import os
import json
import re
import copy
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st      # ← ADD THIS
from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate


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

    # Apply any existing date heuristics
    spec = _augment_spec_with_date_heuristics(question, spec)

    # 🔧 Extra post-processing for patterns that the LLM often misses
    q_lower = (question or "").lower()

    # Always keep the raw question text for downstream helpers
    spec["_question_text"] = question

    # ===== CRITICAL FIX: Force growth_rate for growth-related questions =====
    growth_keywords = [
        "growth rate", "growth rates", "weekly growth", "week-over-week",
        "wow", "yoy", "year-over-year", "yearly growth"
    ]
    if any(kw in q_lower for kw in growth_keywords):
        print(f"🔴 FORCING growth_rate aggregation (detected keywords)")
        spec["aggregation"] = "growth_rate"
        
        # Make sure we have the right group_by for time dimension
        gb = spec.get("group_by") or []
        if "weekly" in q_lower or "week" in q_lower:
            if "week" not in gb:
                gb.append("week")
            print(f"🔴 Added 'week' to group_by: {gb}")
        elif "yearly" in q_lower or "year" in q_lower or "yoy" in q_lower:
            if "year" not in gb:
                gb.append("year")
            print(f"🔴 Added 'year' to group_by: {gb}")
        
        spec["group_by"] = gb

    # -------------------------------
    # Distributor compare: DSD vs PI Medical
    # -------------------------------
    if "compare" in q_lower and "dsd" in q_lower and "pi medical" in q_lower:
        spec["aggregation"] = "compare"
        compare = spec.get("compare") or {}
        compare["entities"] = ["DSD", "PI Medical"]
        compare["entity_type"] = "distributor"
        spec["compare"] = compare

        gb = spec.get("group_by") or []

        if any(t in q_lower for t in ["per year", "by year", "yearly", "each year", "in years"]):
            if "year" not in gb:
                gb.append("year")

        if "distributor" not in gb:
            gb.insert(0, "distributor")

        spec["group_by"] = gb

    # -------------------------------
    # Product compare: CA vs NCA
    # -------------------------------
    if "compare" in q_lower and "ca" in q_lower and "nca" in q_lower:
        spec["aggregation"] = "compare"
        compare = spec.get("compare") or {}
        compare["entities"] = ["CA", "NCA"]
        compare["entity_type"] = "product_sold"
        spec["compare"] = compare

        gb = spec.get("group_by") or []

        if any(t in q_lower for t in ["per year", "by year", "yearly", "each year", "in years"]):
            if "year" not in gb:
                gb.append("year")

        if "product_sold" not in gb:
            gb.insert(0, "product_sold")

        spec["group_by"] = gb

    # -------------------------------
    # Mark "why / reason / drop" style questions
    # -------------------------------
    why_keywords = ["why", "reason", "reasons", "cause", "drop", "decline", "decrease", "went down"]
    is_why = any(k in q_lower for k in why_keywords)

    if is_why:
        spec["_why_question"] = True
        filters = spec.get("filters") or {}
        spec["filters"] = filters

        if filters.get("year"):
            try:
                spec["_why_year"] = int(filters["year"])
            except Exception:
                pass
        if filters.get("week"):
            try:
                spec["_why_week"] = int(filters["week"])
            except Exception:
                pass

        # Fix awkward "all statuses" + shipping-status breakdowns
        if spec.get("shipping_status_mode") == "all":
            spec["shipping_status_mode"] = "countable"
            spec["shipping_status_list"] = []

        gb = spec.get("group_by") or []
        if gb == ["shipping_status"]:
            gb = []
        elif "shipping_status" in gb and len(gb) > 1:
            gb = [g for g in gb if g != "shipping_status"]
        spec["group_by"] = gb

    # -------------------------------
    # Global safety net: "per year" wording
    # -------------------------------
    if any(t in q_lower for t in ["per year", "by year", "yearly", "each year", "in years"]):
        gb = spec.get("group_by") or []
        if "year" not in gb:
            gb.append("year")
        spec["group_by"] = gb

    q_stripped = (question or "").strip().lower()
    if q_stripped.startswith("why"):
        spec["_why_question"] = True
        filters = spec.get("filters") or {}
        spec["filters"] = filters
        if filters.get("year"):
            try:
                spec["_why_year"] = int(filters["year"])
            except Exception:
                pass
        if filters.get("week"):
            try:
                spec["_why_week"] = int(filters["week"])
            except Exception:
                pass

    # 🔧 Hard override for "drop in week X (of Y)" style questions
    filters = spec.get("filters") or {}
    spec["filters"] = filters

    if any(k in q_lower for k in ["drop", "demand"]) and "week" in q_lower:
        m = re.search(r"week\s+(\d{1,2})\s*(?:of\s+)?(20[2-3][0-9])?", q_lower)
        if m:
            week_val = int(m.group(1))
            filters["week"] = week_val

            year_str = m.group(2)
            if year_str:
                filters["year"] = int(year_str)

            spec["_why_question"] = True
            spec["_why_week"] = week_val
            if year_str:
                spec["_why_year"] = int(year_str)

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
    q = (question or "").lower()
    spec = spec or {}

    # Ensure filters exist
    filters = spec.get("filters") or {}
    spec["filters"] = filters

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
    tw = spec.get("time_window") or {
        "mode": None,
        "n_weeks": None,
        "anchor": {"year": None, "week": None},
    }
    if "anchor" not in tw:
        tw["anchor"] = {"year": None, "week": None}
    if "year" not in tw["anchor"]:
        tw["anchor"]["year"] = None
    if "week" not in tw["anchor"]:
        tw["anchor"]["week"] = None

    # Only add our heuristics if LLM did NOT already set a mode
    if not tw.get("mode"):
        # --- last / past / previous / trailing / rolling N weeks ---
        m_last_weeks = re.search(
            r"(last|past|previous|trailing|rolling)\s+(\d+)\s+weeks?",
            q,
        )
        if m_last_weeks:
            n_weeks_str = m_last_weeks.group(2)
            try:
                n_weeks = int(n_weeks_str)
                tw["mode"] = "last_n_weeks"
                tw["n_weeks"] = n_weeks
                tw["anchor"]["year"] = None
                tw["anchor"]["week"] = None
                # IMPORTANT: dynamic window → we must NOT also use filters.week
                filters["week"] = None
            except Exception:
                pass

    # Anchored window: "N weeks before week X [of YEAR]"
    if not tw.get("mode"):
        m_anchor = re.search(
            r"(\d+)\s+weeks?\s+(?:before|leading up to)\s+week\s+(\d+)(?:\s+of\s+(20[2-3][0-9]))?",
            q,
        )
        if m_anchor:
            try:
                n_weeks = int(m_anchor.group(1))
                week_x = int(m_anchor.group(2))
            except Exception:
                n_weeks = None
                week_x = None

            if n_weeks is not None and week_x is not None:
                tw["mode"] = "anchored_last_n_weeks"
                tw["n_weeks"] = n_weeks
                tw["anchor"]["week"] = week_x

                year_str = m_anchor.group(3)
                if year_str:
                    try:
                        year_val = int(year_str)
                        tw["anchor"]["year"] = year_val
                        filters["year"] = year_val
                    except Exception:
                        pass

                # For anchored windows we also clear filters.week
                filters["week"] = None

    spec["filters"] = filters
    spec["time_window"] = tw

    # --- ENSURE WEEK EXTRACTION ALWAYS HAPPENS ---
    # Handles cases like "reason for the drop in week 20"
    if not filters.get("week"):
        m = re.search(r"week\s+(\d{1,2})", q)
        if m:
            try:
                filters["week"] = int(m.group(1))
            except Exception:
                pass

    # --- ENSURE YEAR EXTRACTION ALWAYS HAPPENS ---
    # Handles cases where user omits year but previous question had one
    if not filters.get("year"):
        # 1. Try explicit year in text
        m = re.search(r"\b(20[2-3][0-9])\b", q)
        if m:
            filters["year"] = int(m.group(1))
        else:
            # 2. Infer from history (if available)
            # You must add history support: _augment_spec... currently doesn't receive it
            last_year = spec.get("_inferred_year_from_history")
            if last_year:
                filters["year"] = last_year
            else:
                # 3. Fallback to latest year in metadata
                try:
                    filters["year"] = int(METADATA["Year"].max())
                except Exception:
                    pass

    # --- FINAL: if BOTH year & week exist, enable metadata ---
    spec["_metadata_ready"] = (
        filters.get("year") is not None and
        filters.get("week") is not None
    )
    # --- WEEK (explicit single week if no dynamic time_window) ---
    # If we don't already have a week, and there is no dynamic window,
    # look for patterns like "week 20", "week 3", "week 25 of 2025", etc.
    tw = spec.get("time_window") or {}
    if not filters.get("week") and not tw.get("mode"):
        m = re.search(r"\bweek\s+(\d{1,2})\b", q)
        if m:
            try:
                filters["week"] = int(m.group(1))
            except ValueError:
                pass

    return spec


def _normalize_product_filters(spec: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Ensure we default to using the commercial 'product_sold' dimension
    unless the user clearly asked about what was PRODUCED / MANUFACTURED.
    """
    q = (question or "").lower()
    filters = spec.get("filters", {}) or {}

    prod_sold = filters.get("product_sold")
    prod_cat = filters.get("product_catalogue")

    production_words = [
        "produce", "produced", "production", "manufacture", "manufactured", "batch"
    ]
    is_production_question = any(w in q for w in production_words)

    if prod_sold and prod_cat and not is_production_question:
        filters["product_catalogue"] = None

    if prod_cat and not prod_sold and not is_production_question:
        filters["product_sold"] = prod_cat
        filters["product_catalogue"] = None

    spec["filters"] = filters
    return spec

def _inject_customer_from_question(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to infer a *customer* filter from the question text.
    This is more aggressive - looks for ANY entity name in the data that matches.
    """
    filters = spec.get("filters") or {}
    
    # If the LLM already set a customer, don't override it
    if filters.get("customer"):
        print(f"  Customer already set: {filters['customer']}")
        return spec

    question = (spec.get("_question_text") or "").lower()
    print(f"  Trying to extract customer from: {question}")

    if "Customer" not in df.columns:
        print(f"  'Customer' column not in df")
        return spec

    unique_customers = [str(x).strip() for x in df["Customer"].dropna().unique()]
    print(f"  Unique customers in data: {unique_customers[:5]}...")  # Show first 5
    
    # Try exact match first (case-insensitive)
    for cust in unique_customers:
        cust_lower = cust.lower()
        if cust_lower in question:
            print(f"  ✅ FOUND exact match: {cust}")
            filters["customer"] = cust
            spec["filters"] = filters
            return spec
    
    # Try partial match
    q_tokens = [t.strip() for t in question.replace(",", " ").split() if len(t.strip()) > 2]
    print(f"  Question tokens: {q_tokens}")
    
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
            print(f"    Candidate: {cust} (score: {score})")

    # Require at least 1 token match to avoid random noise
    if best_match and best_score > 0:
        print(f"  ✅ FOUND partial match: {best_match} (score: {best_score})")
        filters["customer"] = best_match
        spec["filters"] = filters
        return spec
    
    print(f"  ❌ No customer match found")
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
    result = df.copy()

    def contains(col_name: str, value: str) -> pd.Series:
        return result[col_name].astype(str).str.contains(
            str(value), case=False, na=False, regex=False
        )

    # Customer
    if filters.get("customer") and mapping.get("customer"):
        result = result[contains(mapping["customer"], filters["customer"])]

    # Distributor
    if filters.get("distributor") and mapping.get("distributor"):
        result = result[contains(mapping["distributor"], filters["distributor"])]

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

    # ❌ IMPORTANT: we removed the old block that did:
    #    entities = compare.get("entities") ...
    #    result = result[result[col_name].isin(entities)]
    # That was wiping out rows for fuzzy entities like "DSD Pharma GmbH" vs "DSD".

    # --- Shipping status filter ---
    ship_mode = spec.get("shipping_status_mode", "countable")
    ship_col = mapping.get("shipping_status")

    if ship_col:
        if ship_mode == "countable":
            # Only "normal" shipped/processing orders
            result = result[result[ship_col].isin(COUNTABLE_STATUSES)]

        elif ship_mode == "cancelled":
            # Only cancellations
            result = result[result[ship_col].isin(CANCELLED_STATUSES)]

        elif ship_mode == "explicit" and spec.get("shipping_status_list"):
            # Explicit list of statuses
            result = result[result[ship_col].isin(spec["shipping_status_list"])]

        elif ship_mode == "all":
            # Do NOT filter by status – keep everything
            pass

    # --- Dynamic week window (last N weeks logic) ---
    time_window = spec.get("time_window") or {}
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
    Calculate year-over-year growth for each entity (distributor, customer, etc.).
    Returns a pivoted table with years as columns and growth rates as values.
    """
    if df is None or df.empty:
        return None, float("nan")
    
    # Get non-year grouping dimensions (e.g., Distributor, Country, etc.)
    entity_cols = [c for c in group_cols if c != year_col]
    
    if not entity_cols:
        # No entity dimension, just years -> simple YoY table
        yearly = df.groupby(year_col, as_index=False)[total_col].sum()
        yearly = yearly.sort_values(year_col)
        
        # Calculate YoY growth
        yearly["YoY_Growth"] = yearly[total_col].pct_change()
        return yearly, float("nan")
    
    # Group by entity + year
    grouped = df.groupby(entity_cols + [year_col], as_index=False)[total_col].sum()
    
    # Pivot: entities as rows, years as columns
    pivot = grouped.pivot(index=entity_cols, columns=year_col, values=total_col)
    pivot = pivot.fillna(0)
    
    # Calculate growth rates for each consecutive year pair
    years = sorted(pivot.columns)
    growth_cols = {}
    
    for i in range(1, len(years)):
        prev_year = years[i-1]
        curr_year = years[i]
        col_name = f"{curr_year}_vs_{prev_year}_Growth"
        
        # Calculate growth rate
        growth_cols[col_name] = pivot.apply(
            lambda row: ((row[curr_year] - row[prev_year]) / row[prev_year] * 100) 
            if row[prev_year] > 0 else float("nan"),
            axis=1
        )
    
    # Build final table: Entity | 2022 | 2023 | 2024 | 2025 | 2023_vs_2022 | 2024_vs_2023 | 2025_vs_2024
    result = pivot.reset_index()
    
    for col_name, growth_series in growth_cols.items():
        result[col_name] = growth_series.values
    
    # Format growth columns as percentages (rounded to 1 decimal)
    for col in result.columns:
        if "_Growth" in str(col):
            result[col] = result[col].round(1)
    
    return result, float("nan")

def _calculate_wow_growth(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    group_cols: List[str],
    week_col: str,
    total_col: str,
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate week-over-week growth for each entity.
    """
    print("\n✅ _calculate_wow_growth() CALLED")
    print(f"  df.shape: {df.shape}")
    print(f"  week_col: {week_col}")
    print(f"  total_col: {total_col}")
    print(f"  group_cols passed: {group_cols}")
    print(f"  df.columns: {list(df.columns)}")
    
    if df is None or df.empty:
        print("  ⚠️ df is empty - returning None")
        return None, float("nan")
    
    if week_col not in df.columns:
        print(f"  ⚠️ {week_col} not in df.columns - returning None")
        return None, float("nan")
    
    if total_col not in df.columns:
        print(f"  ⚠️ {total_col} not in df.columns - returning None")
        return None, float("nan")
    
    # Entity columns are everything EXCEPT the week column
    entity_cols = [c for c in group_cols if c != week_col]
    print(f"  entity_cols: {entity_cols}")
    
    # Case A: No entity dimension (just weeks)
    if not entity_cols:
        print("  → Case A: No entity cols (just weeks)")
        weekly = df.groupby(week_col, as_index=False)[total_col].sum()
        weekly = weekly.sort_values(week_col)
        weekly["WoW_Growth_%"] = weekly[total_col].pct_change() * 100
        weekly["WoW_Growth_%"] = weekly["WoW_Growth_%"].round(1)
        
        weekly.columns = [str(c) for c in weekly.columns]
        weekly = weekly.reset_index(drop=True)
        
        print(f"  Result shape: {weekly.shape}")
        print(f"  Result columns: {list(weekly.columns)}")
        return weekly, float("nan")
    
    # Case B: Entity + weeks
    print("  → Case B: Entity + weeks")
    print(f"    Looking for unique values in: {entity_cols}")
    
    result_rows = []
    
    # Get unique entity combinations
    try:
        unique_entities = df[entity_cols].drop_duplicates().values
        print(f"    Found {len(unique_entities)} unique entities")
    except Exception as e:
        print(f"    ERROR getting unique entities: {e}")
        return None, float("nan")
    
    for entity_vals in unique_entities:
        # Filter to this specific entity
        entity_mask = pd.Series(True, index=df.index)
        for i, col in enumerate(entity_cols):
            entity_mask &= (df[col] == entity_vals[i])
        
        entity_df = df[entity_mask].sort_values(week_col).copy()
        
        if entity_df.empty:
            print(f"    Skipping entity {entity_vals} - no data")
            continue
        
        print(f"    Processing entity {entity_vals}: {len(entity_df)} rows")
        
        # Group by week within this entity
        entity_by_week = entity_df.groupby(week_col, as_index=False)[total_col].sum()
        entity_by_week = entity_by_week.sort_values(week_col)
        
        # Calculate WoW growth
        entity_by_week["WoW_Growth_%"] = entity_by_week[total_col].pct_change() * 100
        
        # Attach entity dimensions back
        for i, col in enumerate(entity_cols):
            entity_by_week[col] = entity_vals[i]
        
        result_rows.append(entity_by_week)
    
    if not result_rows:
        print("  ⚠️ No result rows - returning None")
        return None, float("nan")
    
    print(f"  Concatenating {len(result_rows)} result rows")
    result = pd.concat(result_rows, ignore_index=True)
    result["WoW_Growth_%"] = result["WoW_Growth_%"].round(1)
    
    # Reorder columns: entities first, then week, then metrics
    cols_order = entity_cols + [week_col, total_col, "WoW_Growth_%"]
    result = result[[c for c in cols_order if c in result.columns]]
    
    # Ensure all columns are strings
    result.columns = [str(c) for c in result.columns]
    
    # Reset index
    result = result.reset_index(drop=True)
    
    # Ensure numeric columns
    if week_col in result.columns:
        result[week_col] = pd.to_numeric(result[week_col], errors='coerce').fillna(0).astype(int)
    if total_col in result.columns:
        result[total_col] = pd.to_numeric(result[total_col], errors='coerce').fillna(0).astype(int)
    
    print(f"  Final result shape: {result.shape}")
    print(f"  Final columns: {list(result.columns)}")
    print(f"  First 3 rows:\n{result.head(3)}")
    
    return result, float("nan")
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

def _run_aggregation(
    df_filtered: pd.DataFrame,
    spec: Dict[str, Any],
    full_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Run the specified aggregation (sum, average, share_of_total, growth_rate)
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
    
    debug_msg = f"""
🔴 _run_aggregation() DEBUG:
  aggregation: {aggregation}
  group_by: {group_by}
  group_cols: {group_cols}
  total_col: {total_col}
  base_df.shape: {base_df.shape}
  time_window.mode: {spec.get('time_window', {}).get('mode')}
"""
    
    # ------------------------------------------------------------------
    # GROWTH RATE - NOW ACTUALLY IMPLEMENTED
    # ------------------------------------------------------------------
    if aggregation == "growth_rate":
        debug_msg += "\n✅ GROWTH_RATE aggregation detected\n"
        
        # Determine if this is week-over-week or year-over-year
        time_window = spec.get("time_window") or {}
        compare = spec.get("compare") or {}
        
        debug_msg += f"  time_window.mode: {time_window.get('mode')}\n"
        debug_msg += f"  Case 1 check: time_window.mode in dynamic list = {time_window.get('mode') in ('last_n_weeks', 'anchored_last_n_weeks')}\n"
        
        # Case 1: Dynamic week window (last N weeks, etc.) → WoW
        if time_window.get("mode") in ("last_n_weeks", "anchored_last_n_weeks"):
            debug_msg += "  → MATCHED Case 1: Dynamic week window\n"
            if "Week" in base_df.columns:
                week_col = mapping.get("week")
                debug_msg += f"    week_col: {week_col}, Calling _calculate_wow_growth()\n"
                
                group_df, overall_val = _calculate_wow_growth(
                    base_df, spec, group_cols, week_col, total_col
                )
                debug_msg += f"    Result shape: {group_df.shape if group_df is not None else None}\n"
                st.write(debug_msg)
                return group_df, overall_val
            else:
                debug_msg += f"    'Week' NOT in base_df.columns\n"
        
        # Case 2: Time-to-time comparison (period A vs B) → growth between periods
        period_a = compare.get("period_a")
        period_b = compare.get("period_b")
        if period_a and period_b:
            debug_msg += "  → MATCHED Case 2: Period A vs Period B\n"
            # This would need custom logic to compare two specific periods
            # For now, if we have year comparison, use YoY
            if "Year" in base_df.columns and ("year" in group_by or not group_by):
                year_col = mapping.get("year")
                group_df, overall_val = _calculate_yoy_growth(
                    base_df, spec, group_cols, year_col, total_col
                )
                st.write(debug_msg)
                return group_df, overall_val
        
        # Case 3: Year-over-year (if Year in group_by) → YoY
        if "year" in group_by and "Year" in base_df.columns:
            debug_msg += "  → MATCHED Case 3: Year-over-year\n"
            year_col = mapping.get("year")
            group_df, overall_val = _calculate_yoy_growth(
                base_df, spec, group_cols, year_col, total_col
            )
            st.write(debug_msg)
            return group_df, overall_val
        
        # Case 4: Week-over-week (if Week in group_by) → WoW
        if "week" in group_by and "Week" in base_df.columns:
            debug_msg += "  → MATCHED Case 4: Week in group_by\n"
            week_col = mapping.get("week")
            debug_msg += f"    week_col: {week_col}\n"
            debug_msg += f"    group_cols: {group_cols}\n"
            debug_msg += f"    Calling _calculate_wow_growth()\n"
            
            group_df, overall_val = _calculate_wow_growth(
                base_df, spec, group_cols, week_col, total_col
            )
            debug_msg += f"    Result shape: {group_df.shape if group_df is not None else None}\n"
            st.write(debug_msg)
            return group_df, overall_val
        
        # Fallback: no growth calculation possible
        debug_msg += "  → ⚠️ NO CASE MATCHED FOR GROWTH_RATE\n"
        st.write(debug_msg)
        return None, float("nan")

    # ------------------------------------------------------------------
    # COMPARE MODE (entity or time comparison)
    # ------------------------------------------------------------------
    if aggregation == "compare":
        compare = spec.get("compare") or {}
        entities = compare.get("entities")
        entity_type = compare.get("entity_type")

        if entities and entity_type and entity_type not in ("product_sold", "product_catalogue"):
            entity_col = mapping.get(entity_type)
            if not entity_col or entity_col not in df_filtered.columns:
                return None, float("nan")

            def _match_entity(cell, target):
                if pd.isna(cell):
                    return False
                return str(target).lower() in str(cell).lower()

            rows = []
            for ent in entities:
                sub = df_filtered[df_filtered[entity_col].apply(lambda x: _match_entity(x, ent))]
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["_CompareEntity"] = ent
                rows.append(sub)

            if not rows:
                return None, float("nan")

            df_compare = pd.concat(rows, ignore_index=True)

            extra_group_cols = [
                mapping[g]
                for g in group_by
                if g != entity_type and mapping.get(g)
            ]
            group_cols_with_entity = (
                ["_CompareEntity"] + extra_group_cols
                if extra_group_cols
                else ["_CompareEntity"]
            )

            grouped_df = df_compare.groupby(group_cols_with_entity, as_index=False)[total_col].sum()
            grouped_df[total_col] = grouped_df[total_col].round(0).astype("int64")
            total_mci = float(df_compare[total_col].sum())
            return grouped_df, total_mci

        # Product comparison
        if entities and entity_type in ("product_sold", "product_catalogue"):
            prod_col = (
                mapping.get("product_sold")
                if entity_type == "product_sold"
                else mapping.get("product_catalogue")
            )
            if not prod_col or prod_col not in df_filtered.columns:
                return None, float("nan")

            def _product_subset(df: pd.DataFrame, label: str) -> pd.DataFrame:
                label = str(label).lower()
                series = df[prod_col].astype(str).str.lower()

                nca_like = (
                    series.str.contains("n.c.a", regex=False)
                    | series.str.contains(" nca", regex=False)
                    | series.str.contains("non carrier", regex=False)
                    | series.str.contains("non-carrier", regex=False)
                )

                ca_like_raw = (
                    series.str.contains(" c.a", regex=False)
                    | series.str.contains("(c.a", regex=False)
                    | series.str.contains(" ca ", regex=False)
                    | series.str.contains(" carrier added", regex=False)
                )

                terb_like = (
                    series.str.contains("terb", regex=False)
                    | series.str.contains("tb-161", regex=False)
                    | series.str.contains("tb161", regex=False)
                    | series.str.contains("161tb", regex=False)
                )

                if label == "nca":
                    mask = nca_like & ~terb_like
                    return df[mask]

                if label == "ca":
                    mask = ca_like_raw & ~nca_like & ~terb_like
                    return df[mask]

                return df[series.str.contains(label, na=False)]

            rows: List[pd.DataFrame] = []
            for ent in entities:
                sub = _product_subset(df_filtered, ent)
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["_CompareEntity"] = ent
                rows.append(sub)

            if not rows:
                return None, float("nan")

            df_compare = pd.concat(rows, ignore_index=True)

            extra_group_cols = [
                mapping[g]
                for g in group_by
                if g and mapping.get(g)
            ]
            group_cols_with_entity = (
                ["_CompareEntity"] + extra_group_cols
                if extra_group_cols
                else ["_CompareEntity"]
            )

            grouped_df = df_compare.groupby(
                group_cols_with_entity, as_index=False
            )[total_col].sum()
            grouped_df[total_col] = grouped_df[total_col].round(0).astype("int64")
            total_mci = float(df_compare[total_col].sum())
            return grouped_df, total_mci

        period_a = compare.get("period_a")
        period_b = compare.get("period_b")
        if period_a and period_b:
            spec["aggregation"] = "growth_rate"
            return _run_aggregation(df_filtered, spec, full_df)

        if group_cols:
            grouped_df = df_filtered.groupby(group_cols, as_index=False)[total_col].sum()
            grouped_df[total_col] = grouped_df[total_col].round(0).astype("int64")
            total_mci = float(df_filtered[total_col].sum())
            return grouped_df, total_mci

        total_mci = float(df_filtered[total_col].sum())
        return None, total_mci

    # ------------------------------------------------------------------
    # PROJECTION VS ACTUAL
    # ------------------------------------------------------------------
    if aggregation == "projection_vs_actual":
        actual_col = mapping.get("total_mci")
        proj_col = mapping.get("proj_mci")

        if not actual_col or actual_col not in df_filtered.columns:
            return None, float("nan")
        if not proj_col or proj_col not in df_filtered.columns:
            return None, float("nan")

        proj_keys = [
            c
            for c in [
                "Year",
                "ProjWeek",
                "Week number for Activity vs Projection",
                "Distributor",
            ]
            if c in df_filtered.columns
        ]

        df_actual = df_filtered
        if proj_keys:
            df_proj = df_filtered.drop_duplicates(subset=proj_keys)
        else:
            df_proj = df_filtered

        if group_cols:
            grp_actual = (
                df_actual
                .groupby(group_cols, as_index=False)[actual_col]
                .sum()
                .rename(columns={actual_col: "Actual_mCi"})
            )

            grp_proj = (
                df_proj
                .groupby(group_cols, as_index=False)[proj_col]
                .sum()
                .rename(columns={proj_col: "Projected_mCi"})
            )

            merged = pd.merge(grp_actual, grp_proj, on=group_cols, how="outer")
        else:
            actual_total = float(df_actual[actual_col].sum())
            projected_total = float(df_proj[proj_col].sum())

            merged = pd.DataFrame(
                {
                    "Actual_mCi": [actual_total],
                    "Projected_mCi": [projected_total],
                }
            )

        merged["Actual_mCi"] = merged["Actual_mCi"].fillna(0.0)
        merged["Projected_mCi"] = merged["Projected_mCi"].fillna(0.0)
        merged["Delta_mCi"] = merged["Actual_mCi"] - merged["Projected_mCi"]
        merged["DeltaPct"] = merged.apply(
            lambda r: (r["Delta_mCi"] / r["Projected_mCi"] * 100.0)
            if r["Projected_mCi"] not in (0, 0.0)
            else float("nan"),
            axis=1,
        )

        total_actual = float(merged["Actual_mCi"].sum())
        return merged, total_actual

    # ------------------------------------------------------------------
    # SUM (default)
    # ------------------------------------------------------------------
    if aggregation == "sum_mci":
        total_mci = float(df_filtered[total_col].sum())
        if group_cols:
            grouped_df = df_filtered.groupby(group_cols, as_index=False)[total_col].sum()
            grouped_df[total_col] = grouped_df[total_col].round(0).astype("int64")
            return grouped_df, total_mci
        return None, total_mci

    # ------------------------------------------------------------------
    # AVERAGE
    # ------------------------------------------------------------------
    if aggregation == "average_mci":
        if not group_cols:
            avg_val = float(df_filtered[total_col].mean())
            return None, avg_val

        grouped_df = df_filtered.groupby(group_cols, as_index=False)[total_col].mean()
        grouped_df = grouped_df.rename(columns={total_col: "Average_mCi"})
        overall_avg = float(df_filtered[total_col].mean())
        return grouped_df, overall_avg

    # ------------------------------------------------------------------
    # SHARE OF TOTAL
    # ------------------------------------------------------------------
    if aggregation == "share_of_total":
        if group_cols:
            total = float(df_filtered[total_col].sum())
            if total == 0:
                spec["_share_debug"] = {"denominator": 0.0}
                return None, float("nan")

            grouped_df = df_filtered.groupby(group_cols, as_index=False)[total_col].sum()
            grouped_df["ShareOfTotal"] = grouped_df[total_col] / total
            spec["_share_debug"] = {"denominator": total}
            return grouped_df, float("nan")

        if full_df is None or not isinstance(full_df, pd.DataFrame):
            return None, float("nan")

        numerator = float(df_filtered[total_col].sum())
        base_spec = copy.deepcopy(spec)
        base_filters = base_spec.get("filters", {}) or {}

        for k in ["customer", "distributor", "country", "region"]:
            base_filters[k] = None
        base_spec["filters"] = base_filters

        df_den = _apply_filters(full_df, base_spec)
        if df_den is None or df_den.empty or total_col not in df_den.columns:
            spec["_share_debug"] = {"numerator": numerator, "denominator": 0.0}
            return None, float("nan")

        denominator = float(df_den[total_col].sum())
        spec["_share_debug"] = {"numerator": numerator, "denominator": denominator}

        if denominator == 0:
            return None, float("nan")

        share = numerator / denominator
        return None, float(share)


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

# --------------------------------------------------------------------
# 3) PUBLIC ENTRYPOINT – called from app.py
# --------------------------------------------------------------------
def _disambiguate_customer_vs_distributor(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a best-effort distinction between 'customer' and 'distributor'.

    Rules:
      - If question mentions 'distributor', prefer distributor filter.
      - If question mentions 'customer' (and not 'distributor'), prefer customer filter.
      - If a 'customer' value exactly matches a known distributor name, move it to distributor.
    """
    filters = spec.get("filters") or {}
    question = (spec.get("_question_text") or "").lower()

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

    # 2) No explicit language – try to fix obvious mis-assignments
    #    e.g. 'DSD Pharma GmbH' being set as a customer, but is actually a distributor.
    if "Distributor" in df.columns:
        distributor_names = {str(x).strip().lower() for x in df["Distributor"].dropna().unique()}
    else:
        distributor_names = set()

    if customer_val and not distributor_val:
        cust_norm = str(customer_val).strip().lower()
        if cust_norm in distributor_names:
            # Move it from customer -> distributor
            filters.pop("customer", None)
            filters["distributor"] = customer_val
            spec["filters"] = filters
            return spec

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

def answer_question_from_df(
    question: str,
    consolidated_df: pd.DataFrame,
    history: Optional[List[Dict[str, str]]] = None,
    client=None,
) -> str:
    """Main function to interpret, execute, and answer a question from the DataFrame."""
    if consolidated_df is None or consolidated_df.empty:
        return "The consolidated data is empty. Please load data first."

    print("\n" + "="*70)
    print("DEBUG answer_question_from_df() START")
    print("="*70)
    print(f"Question: {question}")
    
    # 1) Build & normalize the spec
    spec = _interpret_question_with_llm(question, history=history)

    print(f"\nAfter _interpret_question_with_llm():")
    print(f"  aggregation: {spec.get('aggregation')}")
    print(f"  group_by: {spec.get('group_by')}")
    print(f"  filters.customer: {spec.get('filters', {}).get('customer')}")
    print(f"  filters.year: {spec.get('filters', {}).get('year')}")
    
    # 🔧 Ensure we have explicit week/year if they appear in the question text
    q_lower = (question or "").lower()
    filters = spec.get("filters") or {}

    # If LLM didn't set week, try to extract "week 35", "week 7", etc.
    if not filters.get("week"):
        m_week = re.search(r"\bweek\s+(\d{1,2})\b", q_lower)
        if m_week:
            filters["week"] = int(m_week.group(1))
            print(f"  Extracted week from text: {filters['week']}")

    # If LLM didn't set year, try to extract 2024, 2025, etc.
    if not filters.get("year"):
        m_year = re.search(r"\b(20[2-3][0-9])\b", q_lower)
        if m_year:
            filters["year"] = int(m_year.group(1))
            print(f"  Extracted year from text: {filters['year']}")

    spec["filters"] = filters  # keep attached to spec

    # 🔧 Ensure we have a year for "why" questions with an explicit week but no year
    if spec.get("_why_question") and filters.get("week") and not filters.get("year"):
        if "Year" in consolidated_df.columns and not consolidated_df["Year"].dropna().empty:
            filters["year"] = int(consolidated_df["Year"].max())

    spec = _force_cancellation_status_from_text(question, spec)
    spec = _ensure_all_statuses_when_grouped(spec)
    spec = _inject_customer_from_question(consolidated_df, spec)
    spec = _disambiguate_customer_vs_distributor(consolidated_df, spec)

    # 🔧 Final fix for "why / reason / drop" style questions
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

    print(f"\nBefore _apply_filters():")
    print(f"  aggregation: {spec.get('aggregation')}")
    print(f"  group_by: {spec.get('group_by')}")
    
    # 2) Apply filters & run aggregation
    df_filtered = _apply_filters(consolidated_df, spec)
    
    print(f"\nAfter _apply_filters():")
    print(f"  df_filtered.shape: {df_filtered.shape if df_filtered is not None else None}")
    
    print(f"\nCalling _run_aggregation():")
    print(f"  aggregation: {spec.get('aggregation')}")
    
    group_df, numeric_value = _run_aggregation(df_filtered, spec, consolidated_df)

    print(f"\nAfter _run_aggregation():")
    print(f"  group_df is None: {group_df is None}")
    if group_df is not None:
        print(f"  group_df.shape: {group_df.shape}")
        print(f"  group_df.columns: {list(group_df.columns)}")
        print(f"  group_df.dtypes:\n{group_df.dtypes}")
        print(f"  First 3 rows:\n{group_df.head(3)}")
    print(f"  numeric_value: {numeric_value}")
    
    # 3) Pivot-style reshaping for time dimensions
    group_df = _reshape_for_display(group_df, spec)

    # 4) Format numeric columns
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

    # 5) If we have Year + Total_mCi, pivot so years are columns
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
            wide = pd.DataFrame([{
                str(y): float(
                    group_df.loc[group_df["Year"] == y, "Total_mCi"].sum()
                )
                for y in years
            }])
            group_df = wide

    # 6) One more pass of numeric formatting after pivot
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

    # 7) Build human-readable filter description
    filters = spec.get("filters", {}) or {}
    aggregation = spec.get("aggregation", "sum_mci") or "sum_mci"
    
    # ===== DEBUG: Add to core_answer early =====
    spec_debug = f"""
**SPEC DEBUG:**
- aggregation: {aggregation}
- group_by: {spec.get('group_by')}
- filters.customer: {filters.get('customer')}
- filters.year: {filters.get('year')}
- time_window.mode: {spec.get('time_window', {}).get('mode')}
- 'week' in group_by: {'week' in (spec.get('group_by') or [])}
- 'Week' in df_filtered columns: {'Week' in (df_filtered.columns if df_filtered is not None else [])}
"""

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
        status_text = "countable orders (shipped, partially shipped, shipped late, or being processed)"
    elif ship_mode == "cancelled":
        status_text = "cancelled or rejected orders"
    elif ship_mode == "explicit":
        status_text = "orders with the specified shipping statuses"
    else:
        status_text = f"orders with '{ship_mode}' statuses"

    row_count = len(df_filtered) if df_filtered is not None else 0

    # If we have NO table and the numeric value is NaN, it's a real error.
    if group_df is None and (numeric_value is None or pd.isna(numeric_value)):
        return (
            "I couldn't compute the requested metric because there was no valid data "
            "or the base period had zero activity.\n\n" + spec_debug
        )

    # 8) Build the core textual answer by aggregation type
    core_answer = ""

    # SUM (and basic comparison tables)
    if aggregation in ("sum_mci", "compare"):
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
                    f"approximately **{share_pct:.1f}%**, but the denominator could not be fully validated."
                )
        else:
            den = share_debug.get("denominator")
            if den is not None and den != 0:
                denom_txt = f"The total ordered amount (denominator) is **{den:,.0f} mCi**."
            else:
                denom_txt = "The total ordered amount (denominator) is derived from the current filters."
            preview_md = group_df.to_markdown(index=False)
            header = (
                f"Here is the **share of total ordered amount per group** for {status_text} for {filter_text}. "
                f"{denom_txt}\n\n"
            )
            core_answer = header + preview_md

    # GROWTH RATE
    elif aggregation == "growth_rate":
        if group_df is None:
            if numeric_value is None or pd.isna(numeric_value):
                core_answer = (
                    f"I couldn't compute a growth rate for {status_text} for {filter_text} "
                    f"because there was no activity in the base period."
                )
            else:
                pct = numeric_value * 100.0
                core_answer = (
                    f"Based on {status_text} for {filter_text}, the growth rate between the two periods "
                    f"is approximately **{pct:.1f}%**."
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
                        f"Here is the **week-over-week growth** for the last **{n_weeks} weeks** "
                        f"based on {status_text} for {filter_text}.\n\n"
                    )
                else:
                    header = (
                        f"Here is the **week-over-week growth** per group "
                        f"based on {status_text} for {filter_text}.\n\n"
                    )

                core_answer = header + preview_md

            # Year-over-year growth table
            elif "YoY_Growth" in cols:
                preview_md = group_df.to_markdown(index=False)
                header = (
                    f"Here is the **year-over-year growth** per group "
                    f"based on {status_text} for {filter_text}.\n\n"
                )
                core_answer = header + preview_md

            else:
                preview_md = group_df.to_markdown(index=False)
                core_answer = (
                    f"Here is the **growth breakdown per group** for {status_text} for {filter_text}:\n\n"
                    + preview_md
                )

    # Fallback
    else:
        if group_df is None:
            core_answer = (
                f"Based on {status_text} for {filter_text}, the value of the requested metric is "
                f"**{numeric_value:,.0f}** across **{row_count}** rows."
            )
        else:
            preview_md = group_df.to_markdown(index=False)
            core_answer = (
                f"Here is the breakdown of the requested metric for {status_text} for {filter_text}:\n\n"
                + preview_md
            )

    # 9) Optionally add metadata snippet
    meta_block = None
    try:
        if _should_include_metadata(spec):
            meta_block = _build_metadata_snippet(df_filtered, spec)
    except NameError:
        meta_block = None

    # 10) Optionally add chart block
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

    # Optional refinement pass
    try:
        refined_answer = _refine_answer_text(client, final_answer, question)
    except Exception:
        refined_answer = final_answer

    # ===== ADD DEBUG INFO TO RESPONSE =====
    debug_info = f"""
---
**🔴 DEBUG INFO:**
- aggregation: {aggregation}
- group_by: {spec.get('group_by')}
- group_df is None: {group_df is None}
- group_df shape: {group_df.shape if group_df is not None else 'N/A'}
- group_df columns: {list(group_df.columns) if group_df is not None else 'N/A'}
- customer filter: {spec.get('filters', {}).get('customer')}
- year filter: {spec.get('filters', {}).get('year')}
---
"""
    
    refined_answer += debug_info

    print("\n" + "="*70)
    print("DEBUG answer_question_from_df() END")
    print("="*70 + "\n")
    
    return refined_answer
# -------------------------
# Dynamic week window logic
# -------------------------
def _apply_time_window(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies dynamic week window filters based on the 'time_window' structure.
    This handles:
    - last N weeks
    - anchored last N weeks

    Expected structure:
    filters["time_window"] = {
        "mode": "last_n_weeks" | "anchored_last_n_weeks",
        "n_weeks": int,
        "anchor": {
            "year": int | None,
            "week": int | None
        }
    }
    """
    tw = filters.get("time_window") or {}
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
    except Exception:
        return df  # invalid -> ignore gracefully

    df = df.copy()

    # Determine the effective year to operate in
    effective_year = filters.get("year")

    if effective_year is None and anchor_year is not None:
        try:
            effective_year = int(anchor_year)
        except Exception:
            effective_year = None

    if effective_year is None:
        # Fall back to latest year present in the data
        if "Year" not in df.columns or df["Year"].dropna().empty:
            return df
        effective_year = int(df["Year"].max())

    df_year = df[df["Year"] == effective_year]

    if df_year.empty or "Week" not in df_year.columns:
        # No data for that year or no week column -> empty result
        return df.iloc[0:0]

    # Determine anchor week (end of the window)
    if mode == "anchored_last_n_weeks":
        if anchor_week is None:
            return df.iloc[0:0]
        try:
            end_week = int(anchor_week)
        except Exception:
            return df.iloc[0:0]
    else:
        # last_n_weeks -> use the max week of that year
        try:
            end_week = int(df_year["Week"].max())
        except Exception:
            return df.iloc[0:0]

    # Compute start week and clamp to minimum week present
    start_week = end_week - n_weeks + 1
    if start_week < int(df_year["Week"].min()):
        start_week = int(df_year["Week"].min())

    df_filtered = df_year[
        (df_year["Week"] >= start_week) & (df_year["Week"] <= end_week)
    ]

    return df_filtered