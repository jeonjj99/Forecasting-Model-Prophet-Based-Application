# forecast_review.py
# ========================
# Snowflake → POS (weekly→monthly) → merge → compute 3M stats/flags
# ========================

import re
import numpy as np
import pandas as pd
import snowflake.connector
from dataclasses import dataclass
from typing import Dict, Optional

# =========================
# Global parameters
# =========================
RAW_POS_PATH = "WG.xlsx"   # Default POS path (for testing)
RAW_POS_HEADER_ROW = 1
WEEKS_PM = 4.3
CUTOFF = pd.to_datetime("2025-08-01").to_period("M").to_timestamp(how="start")  # Cutoff month (month start)

# Brand ↔ CUSTOMER_KEY mapping (extend as needed)
CUSTOMER_MAP: Dict[str, str] = {
    "WG": "0010002016",  # Walgreens
    "WM": "0010004314",  # Walmart (example)
    "CVS": "0010002009", # CVS (example)
    "DG": "0010007686",  # Dollar General (example)
    "ULTA": "0010008732",
    "FD": "0010007573",
    "TG": "0010003336"
}

# =========================
# Customer-specific Week → Month mapping definition
# =========================
WEEK_DATES_A = [  # CVS, FD, MJ, TG, ULTA, WG
    "01/04","01/11","01/18","01/25","02/01","02/08","02/15","02/22",
    "03/01","03/08","03/15","03/22","03/29","04/05","04/12","04/19",
    "04/26","05/03","05/10","05/17","05/24","05/31","06/07","06/14",
    "06/21","06/28","07/05","07/12","07/19","07/26","08/02","08/09",
    "08/16","08/23","08/30","09/06","09/13","09/20","09/27","10/04",
    "10/11","10/18","10/25","11/01","11/08","11/15","11/22","11/29",
    "12/06","12/13","12/20","12/27"
]
WEEK_DATES_B = [  # WM, DG
    "01-03","01-10","01-17","01-24","01-31","02-07","02-14","02-21","02-28",
    "03-07","03-14","03-21","03-28","04-04","04-11","04-18","04-25","05-02",
    "05-09","05-16","05-23","05-30","06-06","06-13","06-20","06-27","07-04",
    "07-11","07-18","07-25","08-01","08-08","08-15","08-22","08-29","09-05",
    "09-12","09-19","09-26","10-03","10-10","10-17","10-24","10-31","11-07",
    "11-14","11-21","11-28","12-05","12-12","12-19","12-26"
]
WEEK_DATES_BY_BRAND = {
    # Type A
    "CVS": WEEK_DATES_A, "FD": WEEK_DATES_A, "MJ": WEEK_DATES_A,
    "TG": WEEK_DATES_A, "ULTA": WEEK_DATES_A, "WG": WEEK_DATES_A,
    # Type B
    "WM": WEEK_DATES_B, "DG": WEEK_DATES_B,
}

def _build_week_to_month_map_from_brand(brand: str, year_anchor: int = 2022) -> Dict[int, int]:
    dates = WEEK_DATES_BY_BRAND.get(str(brand).upper(), WEEK_DATES_A)
    months = [pd.to_datetime(f"{year_anchor}-{d.replace('/', '-')}", errors="coerce").month for d in dates]
    # week index(1~53) → month(1~12)
    return {i + 1: m for i, m in enumerate(months)}

# =========================
# Snowflake: load forecast (convert MONTH_KEY → YearMonth)
# =========================
def load_fcst_from_snowflake(brand: str = "WG") -> pd.DataFrame:
    user = "JEJEON@KISSUSA.COM"
    account = "UKDVSEA-NPB82638"
    warehouse = "COMPUTE_WH"
    database = "KDB"
    schema = "SCP"

    conn = snowflake.connector.connect(
        user=user,
        account=account,
        authenticator="externalbrowser",
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    cur = conn.cursor()
    cur.execute(f'USE WAREHOUSE {warehouse}')
    cur.execute(f'USE DATABASE {database}')
    cur.execute(f'USE SCHEMA {schema}')

    customer_key = CUSTOMER_MAP.get(brand.upper())
    if not customer_key:
        raise ValueError(f"Unknown brand: {brand}")

    query = f"""
    SELECT "MATERIAL_KEY", "MONTH_KEY", "ESHIP", "ASHIP+Open" AS "ASHIP"
    FROM ZPPRFC01
    WHERE "PLANT_KEY" IN ('G100') AND "CUSTOMER_KEY" = '{customer_key}'
    """
    df_fcst = pd.read_sql(query, conn)

    # MONTH_KEY(YYYYMM) → YearMonth(datetime)
    df_fcst["MONTH_KEY"] = pd.to_datetime(df_fcst["MONTH_KEY"].astype(str), format="%Y%m", errors="coerce")
    df_fcst.rename(columns={"MONTH_KEY": "YearMonth"}, inplace=True)
    # Safely normalize to month start
    df_fcst["YearMonth"] = df_fcst["YearMonth"].dt.to_period("M").dt.to_timestamp(how="start")
    return df_fcst

# =========================
# Load POS & preprocess (aggregate weekly→monthly)
# =========================
def read_pos_raw(path: str, header_row: int = RAW_POS_HEADER_ROW) -> pd.DataFrame:
    return pd.read_excel(path, header=header_row)

def pos_preprocess(pos: pd.DataFrame, brand: str | None = None) -> pd.DataFrame:
    if str(brand).upper() == "TG" and "On-Counter Date" in pos.columns:
        pos = pos[pos["On-Counter Date"].astype(str).str.strip().str.upper() == "STORE"].copy()

    if str(brand).upper() == 'ULTA' and "On-Counter Date" in pos.columns:
        pos = pos[pos["On-Counter Date"].astype(str).str.strip().str.upper() == "IN-STORE"].copy()

    # 1) Remove PPK rows (only for existing status columns)
    def drop_ppk_rows(df: pd.DataFrame,
                      candidate_cols=("SS Status", "FW Status", "Status"),
                      suffix="PPK") -> pd.DataFrame:
        cols = [c for c in candidate_cols if c in df.columns]
        if not cols:
            return df.reset_index(drop=True)
        mask = np.ones(len(df), dtype=bool)
        for c in cols:
            s = df[c].astype(str).str.strip().str.upper()
            mask &= ~s.str.endswith(suffix.upper())
        return df[mask].reset_index(drop=True)

    pos = drop_ppk_rows(pos)

    # 2) Filter to Nail PU
    if "PU" in pos.columns:
        pos = pos[pos["PU"].astype(str).str.strip().str.upper() == "NAIL"]

    # 3) Standardize Segment (only if present)
    if "Segment" in pos.columns:
        pos["Segment"] = pos["Segment"].astype(str).str.strip()
        pos["Segment"].replace({
            "imPRESS ": "imPRESS",
            "Impress": "imPRESS",
            "Preglued Nails": "PreGlued Nails",
            "FRENCH NAILS": "French Nails",
            "Decorated nails": "Decorated Nails",
            "French nails": "French Nails",
            "Color nails": "Color Nails",
            "impress toe nail": "Toe Nails",
        }, inplace=True)
        pos = pos.dropna(subset=["Segment"]).reset_index(drop=True)

    # 4) Select required columns only
    units_cols   = [c for c in pos.columns if re.fullmatch(r"Units Wk\s*\d+", str(c))]
    door_cols    = [c for c in pos.columns if re.fullmatch(r"Door Wk\s*\d+", str(c))]
    instock_cols = [c for c in pos.columns if re.fullmatch(r"Instock % Wk\s*\d+", str(c))]
    base_cols    = [c for c in ["Material", "Segment", "YEAR"] if c in pos.columns]
    pos_sel = pos[base_cols + units_cols + door_cols + instock_cols].copy()

    # 5) Filter to recent years (optional)
    if "YEAR" in pos_sel.columns:
        pos_sel = pos_sel[pos_sel["YEAR"] > 2022]
    return pos_sel

def convert_weekly_to_monthly_long(
    df: pd.DataFrame,
    brand: str = "WG",
    week_to_month_map: Optional[Dict[int, int]] = None,
    instock_scale_if_fraction: float = 100.0
) -> pd.DataFrame:
    """
    Convert weekly (Units/Door/Instock) to monthly aggregation
      - Door (monthly): max
      - Units (monthly): sum
      - Instock (monthly): mean
      - UPM (monthly): simple average of weekly UPM (=Units/Door)
    """
    if week_to_month_map is None:
        week_to_month_map = _build_week_to_month_map_from_brand(brand)

    # 1) Find week columns that actually exist
    def cols_like(prefix: str):
        pat = re.compile(rf"^{re.escape(prefix)}\s*\d+$")
        return [c for c in df.columns if isinstance(c, str) and pat.match(c)]

    units_cols   = cols_like("Units Wk")
    door_cols    = cols_like("Door Wk")
    instock_cols = cols_like("Instock % Wk")

    # 2) id vars
    id_vars = ["Material", "Segment"]
    if "YEAR" in df.columns:
        id_vars += ["YEAR"]

    # 3) Melt + map week → month
    def melt_metric(metric_cols, value_name):
        if not metric_cols:
            return pd.DataFrame(columns=id_vars + ["WeekNum", "Month", value_name])
        long = df[id_vars + metric_cols].melt(id_vars=id_vars, var_name="Week", value_name=value_name)
        long["WeekNum"] = long["Week"].str.extract(r"(\d+)").astype("Int64")
        long["Month"]   = long["WeekNum"].map(week_to_month_map).astype("Int64")
        return long.drop(columns=["Week"])

    u_long = melt_metric(units_cols,   "Units")
    d_long = melt_metric(door_cols,    "Door")
    i_long = melt_metric(instock_cols, "Instock")

    # 4) Merge the three metrics
    key_cols = [c for c in ["Material", "Segment", "YEAR", "WeekNum", "Month"]
                if c in (u_long.columns.union(d_long.columns).union(i_long.columns))]
    merged = u_long.merge(d_long, on=key_cols, how="left").merge(i_long, on=key_cols, how="left")

    # 5) Convert to numeric & rescale Instock (if it comes in 0–1 range, multiply by 100)
    for c in ["Units", "Door", "Instock"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if "Instock" in merged.columns and merged["Instock"].notna().any():
        q95 = merged["Instock"].quantile(0.95)
        if pd.notna(q95) and q95 <= 1.5:
            merged["Instock"] = merged["Instock"] * instock_scale_if_fraction

    # 6) Weekly UPM
    merged["UPM_week"] = np.where(merged.get("Door", 0) > 0,
                                  merged.get("Units", np.nan) / merged.get("Door", np.nan),
                                  np.nan)

    # 7) Create Year/Month/YearMonth
    if "YEAR" in merged.columns:
        merged = merged.rename(columns={"YEAR": "Year"})
    else:
        # If YEAR is missing, fill with CUTOFF year (fallback)
        merged["Year"] = CUTOFF.year
    
    ### Legacy approach
    #merged["Month"]     = merged["Month"].astype(int)
    #merged["Year"]      = merged["Year"].astype(int)
    #merged["YearMonth"] = merged.apply(lambda x: f"{x['Year']}-{int(x['Month']):02d}", axis=1)


    # Safe numeric casting (keep as Int64, then convert to int for to_datetime)
    merged["Month"] = pd.to_numeric(merged["Month"], errors="coerce").astype("Int64")
    merged["Year"]  = pd.to_numeric(merged["Year"],  errors="coerce").astype("Int64")

    # Year/Month → month-start datetime (vectorized)
    ym = pd.to_datetime(
        dict(
            year=merged["Year"].astype(int),
            month=merged["Month"].astype(int),
            day=1
        ),
        errors="coerce"
    )
    merged["YearMonth"] = ym.dt.to_period("M").dt.to_timestamp(how="start")

    # 8) Monthly aggregation
    monthly = (merged.groupby(["Material", "Segment", "Year", "Month", "YearMonth"], as_index=False)
                     .agg(Units=("Units", "sum"),
                          Instock=("Instock", "mean"),
                          Door=("Door", "max"),
                          UPM=("UPM_week", "mean"),
                          Weeks=("UPM_week", "count")))

    # YearMonth → month-start datetime
    monthly["YearMonth"] = pd.to_datetime(monthly["YearMonth"], format="%Y-%m").dt.to_period("M").dt.to_timestamp(how="start")
    return monthly

def add_segment_dummies_and_yearmonth(df: pd.DataFrame) -> pd.DataFrame:
    if "YearMonth" not in df.columns:
        df["YearMonth"] = df.apply(lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1)
        df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m").dt.to_period("M").dt.to_timestamp(how="start")
    if "Segment" in df.columns:
        seg_dum = pd.get_dummies(df["Segment"], prefix="Segment")
        df = pd.concat([df, seg_dum], axis=1)
    return df

def run_pos_preprocessing_pipeline(pos_raw: pd.DataFrame, brand: str = "WG") -> pd.DataFrame:
    pos_sel  = pos_preprocess(pos_raw, brand=brand)
    monthly  = convert_weekly_to_monthly_long(pos_sel, brand=brand)
    monthly  = add_segment_dummies_and_yearmonth(monthly)
    # Expected columns: ['Material','YearMonth','Door','UPM','Instock', ...]
    return monthly

# =========================
# Compute review flags
# =========================
def compute_review_flags(
    df_merged: pd.DataFrame,
    cutoff: pd.Timestamp = CUTOFF,
    flag_thr1: float = 0.30,
    flag_thr2: float = 0.30,
    flag_thr3: float = 0.30,
    compare_months: int = 4,   # ← added: 3 or 4
) -> pd.DataFrame:
    if df_merged.empty:
        return df_merged  # if empty, return as-is

    df = df_merged.copy()
    df["YearMonth"] = pd.to_datetime(df["YearMonth"]).dt.to_period("M").dt.to_timestamp(how="start")
    df["SellOut"]   = df["UPM"] * df["Door"] * WEEKS_PM

    # Validate window (3 or 4)
    if compare_months not in (3, 4):
        compare_months = 4

    # List of the most recent compare_months months (including cutoff)
    months = pd.period_range(
        (cutoff.to_period("M") - (compare_months - 1)),
        cutoff.to_period("M"),
        freq="M"
    ).to_timestamp(how="start")

    df_recent = df.loc[df["YearMonth"].isin(months)].copy()

    # Recent N-month statistics (build suffix dynamically)
    suf = f"{compare_months}M_Avg"
    avg_recent = (
        df_recent.groupby("MATERIAL_KEY", as_index=True)
                 .agg(
                     **{f"ESHIP_{suf}":   ("ESHIP",   "mean")},
                     **{f"ASHIP_{suf}":   ("ASHIP",   "mean")},
                     **{f"SellOut_{suf}": ("SellOut", "mean")},
                     **{f"Instock_{suf}": ("Instock", "mean")},
                     **{f"Door_Range_{suf}": ("Door", lambda x: x.max() - x.min())}
                 )
    )

    # Map stats back into the main dataframe
    for col in avg_recent.columns:
        df[col] = df["MATERIAL_KEY"].map(avg_recent[col])

    # Safe gap computation (handles mixed strings/NaNs)
    def safe_gap(numer, denom):
        numer = pd.to_numeric(numer, errors="coerce")
        denom = pd.to_numeric(denom, errors="coerce")
        return np.where(denom > 0, (numer - denom) / denom, np.nan)

    es_col = f"ESHIP_{suf}"
    as_col = f"ASHIP_{suf}"
    so_col = f"SellOut_{suf}"

    df["Gap1"]  = safe_gap(df[es_col], df[so_col])   # ESHIP vs SellOut
    df["Gap2"]  = safe_gap(df[as_col], df[so_col])   # ASHIP vs SellOut
    df["Gap3"]  = safe_gap(df[es_col], df[as_col])   # ESHIP vs ASHIP

    df["Flag1"] = (np.abs(df["Gap1"]) > flag_thr1).astype("Int64").fillna(0).astype(int)
    df["Flag2"] = (np.abs(df["Gap2"]) > flag_thr2).astype("Int64").fillna(0).astype(int)
    df["Flag3"] = (np.abs(df["Gap3"]) > flag_thr3).astype("Int64").fillna(0).astype(int)
    df["Group"] = df[["Flag1","Flag2","Flag3"]].astype(str).agg("".join, axis=1)

    # Compatibility aliases (so app/downstream can refer to the latest window without renaming)
    df["ESHIP_Recent_Avg"]   = df[es_col]
    df["ASHIP_Recent_Avg"]   = df[as_col]
    df["SellOut_Recent_Avg"] = df[so_col]
    df["Door_Range_Recent"]  = df[f"Door_Range_{suf}"]

    cols_show = [
        "MATERIAL_KEY","Group", "YearMonth","Door","UPM","Instock","SellOut",
        f"Door_Range_{suf}", "Door_Range_Recent",
        "ESHIP", es_col, "ASHIP", as_col, so_col,
        "ESHIP_Recent_Avg","ASHIP_Recent_Avg","SellOut_Recent_Avg",
        "Gap1","Gap2","Gap3","Flag1","Flag2","Flag3"
    ]

    review_month = (df.loc[df["YearMonth"] == cutoff, cols_show]
                      .sort_values(["Flag1","Flag2","Flag3"], ascending=False)
                      .reset_index(drop=True))
    return review_month


# =========================
# Run full pipeline
# =========================
def run_pipeline(pos_path: str = RAW_POS_PATH, brand: str = "WG", cutoff: pd.Timestamp = CUTOFF,
                 flag_thr1: float = 0.30, flag_thr2: float = 0.30, flag_thr3: float = 0.30,
                 compare_months: int = 4) -> pd.DataFrame:

    # 1) Load forecast (MONTH_KEY→YearMonth)
    df_fcst = load_fcst_from_snowflake(brand=brand)

    # 2) Load POS and convert weekly→monthly
    pos_raw = read_pos_raw(pos_path, header_row=RAW_POS_HEADER_ROW)
    monthly = run_pos_preprocessing_pipeline(pos_raw, brand=brand)

    # 3) Keep only alive SKUs (Materials) in the cutoff month
    alive_mats = monthly.loc[
        (monthly["YearMonth"] == cutoff) & (monthly["Door"] > 0),
        "Material"
    ].unique()
    monthly_alive = monthly[monthly["Material"].isin(alive_mats)]

    # 4) Align column names for merging
    monthly_selected = monthly_alive.rename(columns={"Material": "MATERIAL_KEY"}).copy()
    monthly_selected["YearMonth"] = pd.to_datetime(monthly_selected["YearMonth"]).dt.to_period("M").dt.to_timestamp(how="start")
    df_fcst["YearMonth"]          = pd.to_datetime(df_fcst["YearMonth"]).dt.to_period("M").dt.to_timestamp(how="start")

    # 5) Merge
    df_merged = pd.merge(monthly_selected, df_fcst, how="left", on=["MATERIAL_KEY", "YearMonth"])

    # 6) Build review table

    
    review_month = compute_review_flags(
        df_merged, cutoff=cutoff,
        flag_thr1=flag_thr1, flag_thr2=flag_thr2, flag_thr3=flag_thr3,
        compare_months=compare_months
    )
    

    return review_month

# =========================
# CLI test (optional)
# =========================
if __name__ == "__main__":
    # Quick test with WG file path
    out = run_pipeline(RAW_POS_PATH, brand="WG", cutoff=CUTOFF)
    print(out.head(20))
