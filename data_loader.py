import os
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def get_credentials():
    """
    On Streamlit Cloud: use st.secrets['gcp_service_account'] as a dict.
    Locally: fall back to credentials.json file.
    """
    # Streamlit Cloud: TOML table -> dict
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    # Local dev
    service_account_file = os.path.join(SCRIPT_DIR, "credentials.json")
    return Credentials.from_service_account_file(service_account_file, scopes=SCOPES)


creds = get_credentials()
gc = gspread.authorize(creds)


# ---------------------------------------
# Google Sheet IDs
# ---------------------------------------
ORDERS_SHEET_ID = "1WNW4IqAG6W_6XVSaSk9kkukztY93qCDQ3mypZaF5F-Y"
PROJECTION_SHEET_ID = ORDERS_SHEET_ID  # Same file, different tab
METADATA_SHEET_ID = "11GEqq7dR_QjLo7AWzJZRwDyZXsXtZuyOwQkgpwjd9z0"

ORDERS_WORKSHEET_NAME = "Airtable Data"
PROJECTION_WORKSHEET_NAME = "Projection data"
METADATA_WORKSHEET_NAME = "Sheet1"


# ---------------------------------------
# Helpers
# ---------------------------------------
def _open_worksheet(sheet_id: str, worksheet_name: str):
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    return ws


# ---------------------------------------
# Loaders
# ---------------------------------------
def load_orders() -> pd.DataFrame:
    ws = _open_worksheet(ORDERS_SHEET_ID, ORDERS_WORKSHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = [c.strip() for c in df.columns]
    return df


def load_projection() -> pd.DataFrame:
    ws = _open_worksheet(PROJECTION_SHEET_ID, PROJECTION_WORKSHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = [c.strip() for c in df.columns]
    return df


def load_metadata() -> pd.DataFrame:
    ws = _open_worksheet(METADATA_SHEET_ID, METADATA_WORKSHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------
# Preprocessing
# ---------------------------------------
def preprocess_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Orders dataframe is empty")

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

        rename_map = {
        "The customer": "Customer",
        "Total amount ordered (mCi)": "Total_mCi",
        "Year": "Year",
        "Week of supply": "Week",   # fallback only
        "Shipping Status": "ShippingStatus",
        "Catalogue description (sold as)": "Product",
        "Distributing company (from Company name)": "Distributor",
        "Country": "Country",
        "Account Manager Email": "AccountManagerEmail",
    }

    # Apply basic renaming
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # --- CANONICAL WEEK LOGIC ---
    # Always override Week with the Activity-vs-Projection week when available
    if "Week number for Activity vs Projection" in df.columns:
        df["Week"] = df["Week number for Activity vs Projection"]
    # Else fallback: Week stays as renamed from "Week of supply"
    elif "Week" in df.columns:
        pass
    else:
        raise ValueError(
            "No week column found: expected 'Week number for Activity vs Projection' "
            "or 'Week of supply'."
        )

    required = ["Customer", "Total_mCi", "Year", "Week", "ShippingStatus", "Product"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Orders data missing required columns: {missing}")

    df = df.dropna(subset=required)

    for col in ["Total_mCi", "Year", "Week"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Total_mCi", "Year", "Week"])
    df["Year"] = df["Year"].astype(int)
    df["Week"] = df["Week"].astype(int)

    df["YearWeek"] = list(zip(df["Year"], df["Week"]))

    return df