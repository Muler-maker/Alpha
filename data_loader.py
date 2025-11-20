import os

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def get_credentials():
    # Streamlit Cloud: use the table stored in st.secrets["gcp_service_account"]
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    # Local dev: use credentials.json file
    service_account_file = os.path.join(SCRIPT_DIR, "credentials.json")
    return Credentials.from_service_account_file(service_account_file, scopes=SCOPES)


creds = get_credentials()
gc = gspread.authorize(creds)


def get_credentials():
    """
    Use Streamlit secrets on the cloud, and credentials.json locally.
    """
    # On Streamlit Cloud: take the JSON from secrets
    if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
        info = json.loads(st.secrets["gcp_service_account"])
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    # Local dev: use the credentials.json file next to this script
    return Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)


creds = get_credentials()
gc = gspread.authorize(creds)


# --- Sheet IDs & worksheet names ---

# Orders + Projections live in the same spreadsheet
ORDERS_SHEET_ID = "1WNW4IqAG6W_6XVSaSk9kkukztY93qCDQ3mypZaF5F-Y"
PROJECTION_SHEET_ID = ORDERS_SHEET_ID  # same file, different tab
METADATA_SHEET_ID = "11GEqq7dR_QjLo7AWzJZRwDyZXsXtZuyOwQkgpwjd9z0"

ORDERS_WORKSHEET_NAME = "Airtable Data"
PROJECTION_WORKSHEET_NAME = "Projection data"
METADATA_WORKSHEET_NAME = "Sheet1"

# Shipping statuses we care about
# VALID_STATUSES = [
#    "Shipped",
#    "Partially shipped",
#    "Shipped and arrived late",
#   "Order being processed",
#]


def _open_worksheet(sheet_id: str, worksheet_name: str):
    """Internal helper to open a worksheet and return a gspread worksheet."""
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    return ws


def load_orders() -> pd.DataFrame:
    """
    Load raw orders from 'Airtable Data' sheet.
    No heavy preprocessing here; that happens in preprocess_orders().
    """
    ws = _open_worksheet(ORDERS_SHEET_ID, ORDERS_WORKSHEET_NAME)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_projection() -> pd.DataFrame:
    """
    Load raw projection data from 'Projection data' tab.
    """
    ws = _open_worksheet(PROJECTION_SHEET_ID, PROJECTION_WORKSHEET_NAME)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_metadata() -> pd.DataFrame:
    """
    Load raw metadata (Major events database) from 'Sheet1'.
    """
    ws = _open_worksheet(METADATA_SHEET_ID, METADATA_WORKSHEET_NAME)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the core logic you already use in your weekly script:
    - clean column names
    - rename key columns
    - filter by valid shipping statuses
    - convert to proper numeric types
    - create Year/Week ints
    """
    if df is None or df.empty:
        raise ValueError("Orders dataframe is empty")

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "The customer": "Customer",
        "Total amount ordered (mCi)": "Total_mCi",
        "Year": "Year",
        "Week of supply": "Week",
        "Shipping Status": "ShippingStatus",
        "Catalogue description (sold as)": "Product",
        "Distributing company (from Company name)": "Distributor",
        "Country": "Country",
        "Account Manager Email": "AccountManagerEmail",
    }

    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # Filter to valid statuses only
    # if "ShippingStatus" in df.columns:
    #    df = df[df["ShippingStatus"].isin(VALID_STATUSES)]

    # Drop rows missing core fields
    required_cols = ["Customer", "Total_mCi", "Year", "Week", "ShippingStatus", "Product"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Orders data missing required columns: {missing}")

    df = df.dropna(subset=required_cols)

    # Convert numeric fields
    for col in ["Total_mCi", "Year", "Week"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Total_mCi", "Year", "Week"])

    df["Year"] = df["Year"].astype(int)
    df["Week"] = df["Week"].astype(int)

    # Optional: build YearWeek tuple if useful later
    df["YearWeek"] = list(zip(df["Year"], df["Week"]))

    return df
