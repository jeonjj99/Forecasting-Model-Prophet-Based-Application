# FCST_Review_M3_V2.py

# ============================================================

# POS(weekly→monthly) + ASHIP(3M Avg @CUTOFF) → SKU FCST (Long/Short )

# - Group

# - ASHIP Snowflakein load(option) df_fcst possible

# - seasonality UPM (Prophet)

# - [NEW] baseline (USW_baseline × w) + (ASHIP_3M × (1-w)) possible

# - [NEW] Trend: 3month UPM criteria,

# ① consecutive threshold adjustment(mean)

# ② consecutive only (≥ 2×) 70% strength adjustment

# ③ (Default 3month) linear adjustment flat keep

# - key normalization(Material ↔ MATERIAL_KEY) apply

# - precomputed(USWin only seasonality/bucket/split )

# - adjustment " apply " → appin final

# ============================================================


from __future__ import annotations

import re
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
from prophet import Prophet
import snowflake.connector  # df_fcst load


pd.options.mode.copy_on_write = True

# =========================

# Default parameters

# =========================

WEEKS_PM   = 4.3
CUTOFF     = pd.Timestamp("2025-08-01")  # criteriamonth(month)

PERIODS    = 6
MIN_HISTORY = 24
BAND_LOW, BAND_HIGH = 0.851425, 1.148575
NORMALIZE_MEAN_TO_ONE = True
AGG_METHOD_SEG = "mean"      # 'mean'|'median'

MIN_MONTHS_FOR_PROPHET_SEG = 10
RAW_POS_HEADER_ROW = 1
BASELINE_BUFFER = 1
# (Note) If not used 0.70 as fine

TREND_APPLY_FRACTION = 0.70  # needed r2 strength (currently mixed )


# ==== Seasonality blend knobs (for long SKUs) ====

BLEND_ENABLED      = True     # Long SKUin ASHIP seasonality POS seasonality

ASHIP_MIN_HISTORY  = 8        # ASHIP month month ( only ASHIP seasonality )

ALPHA_SEAS         = 0.60     # Long SKU seasonalityin ASHIP (0~1), POS/segment

# BAND_LOW / BAND_HIGH ()



# =========================

# Trend parameters (appin compute_trend_table only )

# - threshold "month " criteria

# =========================

TREND_LOOKBACK = 3            # 3month ( 3month )

TREND_RAMP_MONTHS = 3         # 3month → flat keep

TREND_UP_TOTAL_PCT = 0.20     # month +20% 'uptrend'as

TREND_DOWN_TOTAL_PCT = 0.20   # month -20% 'downtrend'as

TREND_CAP_ABS = 0.50          # |strength| ≤ 0.5 ( ±50%)

TREND_EPS = 1e-8              # 0 prevent


# single spike spike up/spike down adjustment(consecutive): cap 70%only apply

SINGLE_SPIKE_WEIGHT = 0.70
SINGLE_SPIKE_FACTOR = 2.0   # single spike : |r2| >= (threshold × value)

SINGLE_SPIKE_TOL    = 0.05  # (r1) (±)



# ==== NEW: Shortin ASHIP seasonality / ====

SEAS_SHORT_ASHIP_MIN_MONTHS = 14   # Short month ASHIP if present

SEAS_SHORT_BLEND_WEIGHT_USW = 0.35 # Short (segment[USW] ). 0.4 ASHIP 0.6


# =========================

# Week→Month mapping(by brand)

# =========================

WEEK_DATES_A = [
    "01/04","01/11","01/18","01/25","02/01","02/08","02/15","02/22",
    "03/01","03/08","03/15","03/22","03/29","04/05","04/12","04/19",
    "04/26","05/03","05/10","05/17","05/24","05/31","06/07","06/14",
    "06/21","06/28","07/05","07/12","07/19","07/26","08/02","08/09",
    "08/16","08/23","08/30","09/06","09/13","09/20","09/27","10/04",
    "10/11","10/18","10/25","11/01","11/08","11/15","11/22","11/29",
    "12/06","12/13","12/20","12/27"
]
WEEK_DATES_B = [
    "01-03","01-10","01-17","01-24","01-31","02-07","02-14","02-21","02-28",
    "03-07","03-14","03-21","03-28","04-04","04-11","04-18","04-25","05-02",
    "05-09","05-16","05-23","05-30","06-06","06-13","06-20","06-27","07-04",
    "07-11","07-18","07-25","08-01","08-08","08-15","08-22","08-29","09-05",
    "09-12","09-19","09-26","10-03","10-10","10-17","10-24","10-31","11-07",
    "11-14","11-21","11-28","12-05","12-12","12-19","12-26"
]
WEEK_DATES_BY_BRAND = {
    "CVS": WEEK_DATES_A, "FD": WEEK_DATES_A, "MJ": WEEK_DATES_A,
    "TG": WEEK_DATES_A, "ULTA": WEEK_DATES_A, "WG": WEEK_DATES_A,
    "WM": WEEK_DATES_B, "DG": WEEK_DATES_B,
}

def build_week_to_month_map(brand: str, year_anchor: int = 2022) -> Dict[int, int]:
    dates = WEEK_DATES_BY_BRAND.get(str(brand).upper(), WEEK_DATES_A)
    months = [pd.to_datetime(f"{year_anchor}-{d.replace('/', '-')}", errors="coerce").month for d in dates]
    return {i+1: m for i, m in enumerate(months)}

# =========================

# Common utilities

# =========================

def _parse_yearmonth(col: pd.Series) -> pd.Series:
    ds = pd.to_datetime(col, errors="coerce")
    na = ds.isna()
    if na.any():
        ds2 = pd.to_datetime(col[na], format="%b-%y", errors="coerce")
        ds.loc[na] = ds2
    return ds.dt.to_period("M").dt.to_timestamp(how="start")

def _minmax_per_year(s: pd.Series) -> pd.Series:
    vmin, vmax = s.min(), s.max()
    if pd.isna(vmin) or pd.isna(vmax):
        return pd.Series(np.nan, index=s.index, dtype=float)
    if np.isclose(vmax, vmin):
        return pd.Series(0.5, index=s.index, dtype=float)
    return (s - vmin) / (vmax - vmin)

def _linmap(vals: np.ndarray, low: float, high: float) -> np.ndarray:
    if vals.size == 0:
        return np.array([])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmax, vmin):
        return np.full_like(vals, (low + high) / 2.0, dtype=float)
    return low + (vals - vmin) / (vmax - vmin) * (high - low)

def _canon_key_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    return out.str.replace(r"^0+(?!$)", "", regex=True)

def mean_min1(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    total = s.sum(min_count=1)   # NaN NaN Return

    cnt   = s.count()            # NaN count

    if cnt == 0 or pd.isna(total):
        return np.nan
    return total / cnt

# ===== Common utilities additional =====

import unicodedata
import re as _re

SEGMENT_CANON_MAP = {
    "impress": "imPRESS",
    "preglued nails": "PreGlued Nails",
    "french nails": "French Nails",
    "decorated nails": "Decorated Nails",
    "color nails": "Color Nails",
    "toe nails": "Toe Nails",  # Toe Nails standardize

    "impress toe nail": "Toe Nails"
}

def canon_seg_series(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .apply(lambda x: unicodedata.normalize("NFKC", x))   # Unicode normalization

              .str.replace(r"\s+", " ", regex=True)                # multiple inner spaces → single space

              .str.strip()
              .str.lower())

def normalize_segment(s: pd.Series) -> pd.Series:
    # canonical key() standard name mapping

    key = canon_seg_series(s)
    mapped = key.replace(SEGMENT_CANON_MAP)
    return mapped


# =========================

# POS preprocess(weekly→monthly)

# =========================

def read_pos_raw(path: str, header_row: int = RAW_POS_HEADER_ROW) -> pd.DataFrame:
    return pd.read_excel(path, header=header_row)

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

def pos_preprocess(pos: pd.DataFrame, brand: str | None = None) -> pd.DataFrame:
    if str(brand).upper() == "TG" and "On-Counter Date" in pos.columns:
        pos = pos[pos["On-Counter Date"].astype(str).str.strip().str.upper() == "STORE"].copy()
    if str(brand).upper() == 'ULTA' and "On-Counter Date" in pos.columns:
        pos = pos[pos["On-Counter Date"].astype(str).str.strip().str.upper() == "IN-STORE"].copy()
    pos = drop_ppk_rows(pos)
    if "PU" in pos.columns:
        pos = pos[pos["PU"].astype(str).str.strip().str.upper() == "NAIL"]
    if "Segment" in pos.columns:
        pos["Segment"] = normalize_segment(pos["Segment"])
        # display standard name( )as

        pos["Segment"] = pos["Segment"].replace(SEGMENT_CANON_MAP)
        pos = pos.dropna(subset=["Segment"]).reset_index(drop=True)


    units_cols   = [c for c in pos.columns if re.fullmatch(r"Units Wk\s*\d+", str(c))] or []
    door_cols    = [c for c in pos.columns if re.fullmatch(r"Door Wk\s*\d+", str(c))] or []
    instock_cols = [c for c in pos.columns if re.fullmatch(r"Instock % Wk\s*\d+", str(c))] or []
    base_cols    = [c for c in ["Material","Segment","YEAR"] if c in pos.columns]
    pos_sel = pos[base_cols + units_cols + door_cols + instock_cols].copy()

    if "YEAR" in pos_sel.columns:
        pos_sel = pos_sel[pos_sel["YEAR"] > 2022]
    return pos_sel

def convert_weekly_to_monthly_long(
    df: pd.DataFrame,
    brand: str,
    instock_scale_if_fraction: float = 100.0,
    cutoff_for_missing_year: pd.Timestamp = CUTOFF,
) -> pd.DataFrame:
    wmap = build_week_to_month_map(brand)

    def cols_like(prefix: str):
        pat = re.compile(rf"^{re.escape(prefix)}\s*\d+$")
        return [c for c in df.columns if isinstance(c, str) and pat.match(c)]

    units_cols   = cols_like("Units Wk")
    door_cols    = cols_like("Door Wk")
    instock_cols = cols_like("Instock % Wk")

    id_vars = ["Material","Segment"] + (["YEAR"] if "YEAR" in df.columns else [])

    def melt_metric(metric_cols, value_name):
        if not metric_cols:
            return pd.DataFrame(columns=id_vars + ["WeekNum","Month",value_name])
        long = df[id_vars + metric_cols].melt(id_vars=id_vars, var_name="Week", value_name=value_name)
        long["WeekNum"] = long["Week"].str.extract(r"(\d+)").astype("Int64")
        long["Month"]   = long["WeekNum"].map(wmap).astype("Int64")
        return long.drop(columns=["Week"])

    u_long = melt_metric(units_cols,   "Units")
    d_long = melt_metric(door_cols,    "Door")
    i_long = melt_metric(instock_cols, "Instock")

    present = set(u_long.columns) | set(d_long.columns) | set(i_long.columns)
    key_cols = [c for c in ["Material","Segment","YEAR","WeekNum","Month"] if c in present]
    merged = u_long.merge(d_long, on=key_cols, how="left").merge(i_long, on=key_cols, how="left")

    for c in ["Units","Door","Instock"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if "Instock" in merged.columns and merged["Instock"].notna().any():
        if merged["Instock"].quantile(0.95) <= 1.5:
            merged["Instock"] = merged["Instock"] * instock_scale_if_fraction

    merged["UPM_week"] = np.where(merged.get("Door", 0) > 0,
                                  merged.get("Units", np.nan) / merged.get("Door", np.nan),
                                  np.nan)
    if "YEAR" in merged.columns:
        merged = merged.rename(columns={"YEAR":"Year"})
    else:
        merged["Year"] = cutoff_for_missing_year.year

    merged["Month"] = pd.to_numeric(merged["Month"], errors="coerce").astype("Int64")
    merged["Year"]  = pd.to_numeric(merged["Year"],  errors="coerce").astype("Int64")
    ym = pd.to_datetime(dict(year=merged["Year"].astype(int), month=merged["Month"].astype(int), day=1), errors="coerce")
    merged["YearMonth"] = ym.dt.to_period("M").dt.to_timestamp(how="start")

    monthly = (merged.groupby(["Material","Segment","Year","Month","YearMonth"], as_index=False)
               .agg(Units=("Units","sum"),
                    Instock=("Instock","mean"),
                    Door=("Door","max"),
                    UPM=("UPM_week","mean"),
                    Weeks=("UPM_week","count")))
    monthly["YearMonth"] = pd.to_datetime(monthly["YearMonth"]).dt.to_period("M").dt.to_timestamp(how="start")
    return monthly

def add_segment_dummies(df: pd.DataFrame) -> pd.DataFrame:
    if "Segment" in df.columns:
        seg_dum = pd.get_dummies(df["Segment"], prefix="Segment")
        df = pd.concat([df, seg_dum], axis=1)
    return df

# =========================

# Door@CUTOFF

# =========================

def build_door_base(df_all: pd.DataFrame, cutoff: pd.Timestamp) -> Tuple[pd.DataFrame, str]:
    cutoff_ts = pd.Timestamp(cutoff).to_period("M").to_timestamp(how="start")
    DOOR_COL  = f"Door_{cutoff_ts:%Y_%m}"

    df = df_all.copy()
    df["YearMonth"] = _parse_yearmonth(df["YearMonth"])
    df["Door"] = pd.to_numeric(df.get("Door", np.nan), errors="coerce")

    door_exact = (df.loc[df["YearMonth"] == cutoff_ts, ["Material","Door"]]
                    .rename(columns={"Door":"Door_exact"}))
    door_month = (df.loc[(df["YearMonth"].dt.year==cutoff_ts.year)&(df["YearMonth"].dt.month==cutoff_ts.month),
                         ["Material","YearMonth","Door"]]
                    .sort_values(["Material","YearMonth"])
                    .groupby("Material", as_index=False).last()[["Material","Door"]]
                    .rename(columns={"Door":"Door_month"}))
    door_le = (df.loc[(df["YearMonth"]<=cutoff_ts)&(df["YearMonth"].dt.year==cutoff_ts.year),
                      ["Material","YearMonth","Door"]]
                 .sort_values(["Material","YearMonth"])
                 .groupby("Material", as_index=False).last()[["Material","Door"]]
                 .rename(columns={"Door":"Door_le"}))

    base = door_exact.merge(door_month, on="Material", how="outer").merge(door_le, on="Material", how="outer")
    base[DOOR_COL] = base["Door_exact"].fillna(base["Door_month"]).fillna(base["Door_le"])
    return base[["Material", DOOR_COL]].drop_duplicates("Material"), DOOR_COL



# =========================

# ASHIP load & baseline(3M mean)

# =========================

CUSTOMER_MAP: Dict[str, str] = {
    "WG": "0010002016", "WM": "0010004314", "CVS": "0010002009",
    "DG": "0010007686", "ULTA": "0010008732", "FD": "0010007573", "TG": "0010003336"
}

def load_fcst_from_snowflake(brand: str) -> pd.DataFrame:
    user = "JEJEON@KISSUSA.COM"
    account = "UKDVSEA-NPB82638"
    warehouse = "COMPUTE_WH"
    database = "KDB"
    schema = "SCP"
    conn = snowflake.connector.connect(
        user=user, account=account, authenticator="externalbrowser",
        warehouse=warehouse, database=database, schema=schema
    )
    cur = conn.cursor()
    cur.execute(f'USE WAREHOUSE {warehouse}')
    cur.execute(f'USE DATABASE {database}')
    cur.execute(f'USE SCHEMA {schema}')

    customer_key = CUSTOMER_MAP.get(brand.upper())
    if not customer_key:
        raise ValueError(f"Unknown brand: {brand}")

    q = f"""
    SELECT "MATERIAL_KEY", "MONTH_KEY", "ASHIP+Open" AS "ASHIP"
    FROM ZPPRFC01
    WHERE "PLANT_KEY" IN ('G100') AND "CUSTOMER_KEY" = '{customer_key}'
    """
    df = pd.read_sql(q, conn)

    df["MONTH_KEY"] = pd.to_datetime(df["MONTH_KEY"].astype(str), format="%Y%m", errors="coerce")
    df = df.rename(columns={"MONTH_KEY":"YearMonth"})
    df["YearMonth"] = df["YearMonth"].dt.to_period("M").dt.to_timestamp(how="start")
    df["MATERIAL_KEY"] = _canon_key_series(df["MATERIAL_KEY"])
    return df[["MATERIAL_KEY","YearMonth","ASHIP"]]

def compute_aship_baseline_4m(df_fcst: pd.DataFrame, cutoff: pd.Timestamp,
                              strict_three_months: bool = False) -> pd.DataFrame:
    df = df_fcst.copy()
    df["MATERIAL_KEY"] = _canon_key_series(df["MATERIAL_KEY"])
    df["YearMonth"] = pd.to_datetime(df["YearMonth"]).dt.to_period("M").dt.to_timestamp(how="start")

    # 3M → 4M

    months = pd.period_range((cutoff - pd.DateOffset(months=3)).to_period("M"),
                             cutoff.to_period("M"),
                             freq="M").to_timestamp(how="start")
    recent = df.loc[df["YearMonth"].isin(months), ["MATERIAL_KEY","YearMonth","ASHIP"]].copy()
    recent["ASHIP"] = pd.to_numeric(recent["ASHIP"], errors="coerce").fillna(0.0)

    # strict_three_months only, keep

    if not strict_three_months:
        return (recent.groupby("MATERIAL_KEY", as_index=False)
                      .agg(baseline_units=("ASHIP","mean")))

    pivot = (recent.pivot_table(index="MATERIAL_KEY", columns="YearMonth", values="ASHIP", aggfunc="mean")
                    .reindex(columns=months, fill_value=0.0))
    return pivot.mean(axis=1).rename("baseline_units").reset_index()


# =========================

# Seasonality (UPM )

# =========================

def build_long_seasonality(df_all_cut: pd.DataFrame, long_list: list[str]) -> pd.DataFrame:
    """
    Long SKU seasonality (M2 Prophet )
      - USW(=UPM_norm ) : Prophet yearly( multiplicative ) → month mean → mean=1 normalize
      - ASHIP : ASHIP_norm(within-year min-max) → Prophet yearly → month mean → mean=1 normalize
      - final: blended = w*USW + (1-w)*ASHIP (w = SEAS_BLEND_WEIGHT_USW)
      - ASHIP data insufficient/failure USW
      - Toe Nailsonly BAND (0.7, 1.3), otherwise (BAND_LOW, BAND_HIGH)
    Return: ["Material","Month","factor"(= final)]
    """

    # ---- parameters (if missing Default) ----

    WEIGHT_USW        = float(globals().get("SEAS_BLEND_WEIGHT_USW", 0.35))
    LONG_MIN_MONTHS   = int(globals().get("SEAS_LONG_MIN_MONTHS", 24))
    ASHIP_MIN_HISTORY = int(globals().get("SEAS_ASHIP_MIN_HISTORY", 6))

    work = df_all_cut.copy()
    work["Material"]  = work["Material"].astype(str).str.strip()
    work["YearMonth"] = pd.to_datetime(work["YearMonth"], errors="coerce")
    work = work.dropna(subset=["Material","YearMonth"]).copy()

    # segment (Toe Nails exception)

    seg_map = (work[["Material","Segment"]].dropna().drop_duplicates("Material").copy())
    seg_map["seg_key"] = seg_map["Segment"].astype(str).str.strip().str.lower()
    seg_lookup = dict(zip(seg_map["Material"], seg_map["seg_key"]))

    rows = []
    long_set = set(str(x) for x in long_list)

    for mat, g0 in work.groupby("Material", sort=False):
        if mat not in long_set:
            continue

        g = g0.sort_values("YearMonth").copy()
        g["ds"]    = g["YearMonth"]
        g["YEAR"]  = g["ds"].dt.year
        g["Month"] = g["ds"].dt.month

        # ----- (1) USW (M2 ) -----

        g["UPM"] = pd.to_numeric(g["UPM"], errors="coerce")
        g["UPM_norm"] = g.groupby("YEAR")["UPM"].transform(_minmax_per_year)
        train_usw = g[["ds","UPM_norm"]].dropna().rename(columns={"UPM_norm":"y"})
        train_usw["y"] = np.clip(train_usw["y"].astype(float), 1e-9, None) 
        # Long and Prophet fit/train possible check

        if train_usw["ds"].nunique() < max(MIN_HISTORY, LONG_MIN_MONTHS):
            # Long criteria below threshold skip

            continue

        m_usw = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            seasonality_mode="multiplicative", interval_width=0.95
        )
        m_usw.fit(train_usw)
        comp_usw = m_usw.predict(train_usw[["ds"]])[["ds","yearly"]].copy()
        comp_usw["factor_raw"] = 1.0 + comp_usw["yearly"]
        comp_usw["Month"] = comp_usw["ds"].dt.month
        usw_month = (comp_usw.groupby("Month", as_index=False)["factor_raw"].mean().sort_values("Month"))
        mu_usw = usw_month["factor_raw"].mean()
        usw_month["usw_factor_norm"] = (usw_month["factor_raw"] / mu_usw) if (pd.notna(mu_usw) and not np.isclose(mu_usw,0)) else usw_month["factor_raw"]

        # ----- (2) ASHIP (Prophet, within-year min-max fit/train) -----

        ash_month = None
        if "ASHIP" in g.columns:
            g["ASHIP"] = pd.to_numeric(g["ASHIP"], errors="coerce")
            # valid count verify

            if g["ASHIP"].notna().sum() >= ASHIP_MIN_HISTORY:
                g["ASHIP_norm"] = g.groupby("YEAR")["ASHIP"].transform(_minmax_per_year)
                train_a = g[["ds","ASHIP_norm"]].dropna().rename(columns={"ASHIP_norm":"y"})
                train_a["y"] = np.clip(train_a["y"].astype(float), 1e-9, None)  
                if train_a["ds"].nunique() >= ASHIP_MIN_HISTORY:
                    m_a = Prophet(
                        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        seasonality_mode="multiplicative", interval_width=0.95
                    )
                    m_a.fit(train_a)
                    comp_a = m_a.predict(train_a[["ds"]])[["ds","yearly"]].copy()
                    comp_a["factor_raw"] = 1.0 + comp_a["yearly"]
                    comp_a["Month"] = comp_a["ds"].dt.month
                    ash_month = (comp_a.groupby("Month", as_index=False)["factor_raw"].mean().sort_values("Month"))
                    mu_a = ash_month["factor_raw"].mean()
                    ash_month["aship_factor_norm"] = (ash_month["factor_raw"] / mu_a) if (pd.notna(mu_a) and not np.isclose(mu_a,0)) else ash_month["factor_raw"]

        # ----- (3) month & -----

        month_avg = usw_month[["Month","usw_factor_norm"]].copy()
        if ash_month is not None:
            month_avg = month_avg.merge(ash_month[["Month","aship_factor_norm"]], on="Month", how="left")

        if "aship_factor_norm" in month_avg.columns and month_avg["aship_factor_norm"].notna().any():
            usw = month_avg["usw_factor_norm"].astype(float)
            ash = month_avg["aship_factor_norm"].astype(float)
            month_avg["blended_norm"] = np.where(
                ash.notna(),
                WEIGHT_USW*usw + (1.0-WEIGHT_USW)*ash,
                usw
            )
        else:
            month_avg["blended_norm"] = month_avg["usw_factor_norm"].astype(float)

        # ----- (4) BAND mapping (Toe Nails exception) -----

        seg_key = seg_lookup.get(mat, "")
        if seg_key == "toe nails":
            low, high = 0.6, 1.4
        elif seg_key == "impress toe nail":
            low, high = 0.6, 1.4
        else:
            low  = float(globals().get("BAND_LOW", 0.851425))
            high = float(globals().get("BAND_HIGH", 1.148575))

        mapped = _linmap(month_avg["blended_norm"].to_numpy(dtype=float), low, high)


        out = month_avg[["Month"]].copy()
        out.insert(0, "Material", mat)
        out["factor"] = mapped
        out["factor_lower"] = out["factor"]
        out["factor_upper"] = out["factor"]
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["Material","Month","factor","factor_lower","factor_upper"])
    return pd.concat(rows, ignore_index=True)[["Material","Month","factor","factor_lower","factor_upper"]]


def build_segment_seasonality(df_all_cut: pd.DataFrame) -> pd.DataFrame:
    work = df_all_cut.copy()

    # Segment adjustment: Segment if missing → mode (most frequent) Segment restore

    if "Segment" not in work.columns:
        seg_dummy_cols = [c for c in work.columns if str(c).startswith("Segment_")]
        if not seg_dummy_cols:
            raise ValueError("Segment 컬럼 또는 Segment_* 더미가 필요합니다.")
        sums = work.groupby("Material")[seg_dummy_cols].sum()
        seg_choice = sums.idxmax(axis=1).to_frame(name="Segment")
        seg_choice["Segment"] = seg_choice["Segment"].str.replace("^Segment_", "", regex=True)
        work = work.merge(seg_choice.reset_index(), on="Material", how="left")

    # within-year minmax normalize UPMas segment only

    work["YEAR"] = work["YearMonth"].dt.year
    work["UPM"]  = pd.to_numeric(work["UPM"], errors="coerce")
    work["UPM_norm"] = work.groupby(["Segment","YEAR"])["UPM"].transform(_minmax_per_year)

    agg = {"mean": "mean", "median": "median"}[AGG_METHOD_SEG]
    seg_ts = (
        work.groupby(["Segment","YearMonth"], as_index=False)
            .agg(y=("UPM_norm", agg))
            .rename(columns={"YearMonth":"ds"})
            .sort_values(["Segment","ds"])
    )

    rows = []
    for seg, g in seg_ts.groupby("Segment", sort=False):
        g = g.dropna(subset=["y"]).sort_values("ds")
        if g.empty:
            continue

        use_prophet = (g["ds"].nunique() >= max(MIN_MONTHS_FOR_PROPHET_SEG, 10))

        if use_prophet:
            # Prophet path: yearly → factor_raw = 1 + yearly

            train = g[["ds","y"]].rename(columns={"y":"y"}).copy()
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                interval_width=0.95,
            )
            m.fit(train)
            comp = m.predict(g[["ds"]])[["ds","yearly"]].copy()
            comp["factor_raw"] = 1.0 + comp["yearly"]
        else:
            # Fallback: month y mean → mean=1 normalize → month mappingas comp

            tmp = g.copy()
            tmp["Month"] = tmp["ds"].dt.month
            mo = tmp.groupby("Month")["y"].mean()
            # mean=1 normalize ( 0 prevent)

            mo = mo / (mo.mean() if pd.notna(mo.mean()) and not np.isclose(mo.mean(), 0) else 1.0)
            comp = tmp[["ds","Month"]].copy()
            comp["factor_raw"] = comp["Month"].map(mo).astype(float)

        # Month mean → linearmapping(Toe Nails exception)

        comp["Month"] = comp["ds"].dt.month
        month_avg = comp.groupby("Month", as_index=False)["factor_raw"].mean().sort_values("Month")

        if NORMALIZE_MEAN_TO_ONE:
            mu = month_avg["factor_raw"].mean()
            month_avg["factor_norm"] = (
                month_avg["factor_raw"] / mu if (pd.notna(mu) and not np.isclose(mu, 0)) else month_avg["factor_raw"]
            )
        else:
            month_avg["factor_norm"] = month_avg["factor_raw"]

        seg_key = str(seg).strip().lower()
        if seg_key == "toe nails":
            low, high = 0.6, 1.4
        elif seg_key == "impress toe nail":
            low, high = 0.6, 1.4            
        else:
            low, high = BAND_LOW, BAND_HIGH

        month_avg["factor"] = _linmap(month_avg["factor_norm"].to_numpy(float), low, high)

        out = month_avg[["Month","factor"]].copy()
        out.insert(0, "Segment", seg)
        out["factor_lower"] = out["factor"]
        out["factor_upper"] = out["factor"]
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["Segment","Month","factor","factor_lower","factor_upper"])
    return pd.concat(rows, ignore_index=True)[["Segment","Month","factor","factor_lower","factor_upper"]]

def build_short_aship_seasonality(df_all_cut: pd.DataFrame, short_list: list) -> pd.DataFrame:
    work = df_all_cut.copy()
    work["Material"]  = work["Material"].astype(str).str.strip()
    work["YearMonth"] = pd.to_datetime(work["YearMonth"], errors="coerce")
    work = work.dropna(subset=["Material","YearMonth"]).copy()

    # Segment key for band exception (toe/impress toe nail)

    seg_map = (work[["Material","Segment"]].dropna().drop_duplicates("Material").copy())
    seg_key = dict(zip(seg_map["Material"], seg_map["Segment"].astype(str).str.strip().str.lower()))

    rows = []
    short_set = set(str(x) for x in short_list)
    for mat, g0 in work.groupby("Material", sort=False):
        if mat not in short_set:
            continue
        g = g0.sort_values("YearMonth").copy()
        if "ASHIP" not in g.columns:
            continue

        g["ASHIP"] = pd.to_numeric(g["ASHIP"], errors="coerce")
        # valid ASHIP month check

        if g["ASHIP"].notna().sum() < int(globals().get("SEAS_SHORT_ASHIP_MIN_MONTHS", 14)):
            continue

        g["YEAR"]  = g["YearMonth"].dt.year
        g["Month"] = g["YearMonth"].dt.month
        # within-year min-max normalize

        g["ASHIP_norm"] = g.groupby("YEAR")["ASHIP"].transform(_minmax_per_year)

        tmp = g.dropna(subset=["ASHIP_norm"]).copy()
        if tmp.empty:
            continue

        mo = (tmp.groupby("Month")["ASHIP_norm"].mean().sort_index())
        mu = mo.mean()
        # mean=1 normalize

        mo_norm = mo / (mu if (pd.notna(mu) and not np.isclose(mu, 0)) else 1.0)

        # BAND mapping (Toe Nails exception)

        seg = str(seg_key.get(mat, "")).lower()
        if seg in ("toe nails", "impress toe nail"):
            low, high = 0.6, 1.4
        else:
            low  = float(globals().get("BAND_LOW", 0.851425))
            high = float(globals().get("BAND_HIGH", 1.148575))

        mapped = _linmap(mo_norm.to_numpy(dtype=float), low, high)

        out = pd.DataFrame({"Material": mat,
                            "Month": mo.index.values,
                            "aship_short_factor": mapped})
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["Material","Month","aship_short_factor"])
    return pd.concat(rows, ignore_index=True)

# =========================

# (NEW) — 3month, consecutive/single spike mixed rules

# =========================

def _compute_trend_strength_mixed_last3(df_cut: pd.DataFrame,
                                        up_pct: float,
                                        down_pct: float,
                                        cap: float) -> pd.DataFrame:
    """
    3month(t-2, t-1, t) UPMas month r1, r2 :
      r1 = UPM_{t-1}/UPM_{t-2} - 1
      r2 = UPM_{t}  /UPM_{t-1} - 1

    rules:
      ① (r1 >= up_pct) and (r2 >= up_pct) → uptrend: strength = clip(avg(r1, r2), ±cap)
      ② (r1 <= -down_pct) and (r2 <= -down_pct) → downtrend: strength = clip(avg(r1, r2), ±cap)
      ③ only, |r2| >= 2× r1
         - r2 >=  2*up_pct   and r1 > -0.05  → +SINGLE_SPIKE_WEIGHT * cap
         - r2 <= -2*down_pct and r1 <  0.05  → -SINGLE_SPIKE_WEIGHT * cap
      ④ otherwise strength = 0
    """
    work = df_cut.sort_values(["Material","YearMonth"]).copy()
    work["UPM"] = pd.to_numeric(work["UPM"], errors="coerce")

    rows = []
    for mat, g in work.groupby("Material", sort=False):
        g = g.dropna(subset=["UPM"]).sort_values("YearMonth")
        if g.shape[0] < 3:
            rows.append((mat, 0.0)); continue

        u = g["UPM"].to_numpy(float)[-3:]
        if np.any(u <= TREND_EPS):
            rows.append((mat, 0.0)); continue

        r1 = (u[1] / max(u[0], TREND_EPS)) - 1.0
        r2 = (u[2] / max(u[1], TREND_EPS)) - 1.0

        strength = 0.0
        if (r1 >= up_pct) and (r2 >= up_pct):
            strength = np.mean([r1, r2])
        elif (r1 <= -down_pct) and (r2 <= -down_pct):
            strength = np.mean([r1, r2])  # mean → downtrend

        else:
            #if (r2 >= SINGLE_SPIKE_FACTOR*up_pct) and (r1 > -SINGLE_SPIKE_TOL):

            #    strength = +SINGLE_SPIKE_WEIGHT * cap

            #elif (r2 <= -SINGLE_SPIKE_FACTOR*down_pct) and (r1 <  SINGLE_SPIKE_TOL):

            #    strength = -SINGLE_SPIKE_WEIGHT * cap

            
                # r2 apply( cap )

            if (r2 >= SINGLE_SPIKE_FACTOR*up_pct) and (r1 > -SINGLE_SPIKE_TOL):
                strength = float(np.clip(SINGLE_SPIKE_WEIGHT * r2, -cap, cap))
            elif (r2 <= -SINGLE_SPIKE_FACTOR*down_pct) and (r1 <  SINGLE_SPIKE_TOL):
                strength = float(np.clip(SINGLE_SPIKE_WEIGHT * r2, -cap, cap))

            else:
                strength = 0.0

        rows.append((mat, float(np.clip(strength, -cap, cap))))

    return pd.DataFrame(rows, columns=["Material","trend_strength"])

def _build_trend_multiplier_calendar(materials: List[str],
                                     cutoff: pd.Timestamp,
                                     periods: int,
                                     strength_map: pd.DataFrame,
                                     ramp_months: int) -> pd.DataFrame:
    """
    horizon h=1..periods:
      trend_multiplier(h) = 1 + strength * min(h/ramp, 1.0)
    flat( keep).
    """
    future_idx = pd.date_range(start=cutoff + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    cal = pd.MultiIndex.from_product([sorted(materials), future_idx], names=["Material","ds"]).to_frame(index=False)
    cal["h"] = cal.groupby("Material")["ds"].cumcount() + 1
    cal = cal.merge(strength_map, on="Material", how="left")
    cal["trend_strength"] = cal["trend_strength"].fillna(0.0)

    #ramp = max(1, int(ramp_months))

    #ramp_frac = np.minimum(cal["h"] / ramp, 1.0)

    #cal["trend_multiplier"] = 1.0 + cal["trend_strength"] * ramp_frac


    ramp = max(1, int(ramp_months))
    t = np.minimum(cal["h"] / ramp, 1.0).astype(float)

    # ease-out : , only

    progress = 1.0 - (1.0 - t) ** 3  # needed 2→3as

    END_SCALE = 0.8
    progress *= END_SCALE  


    # ⬇️ (: 4month) 2month (=0.50) flat keep

    plateau = 0.50 * END_SCALE
    progress = np.where(cal["h"] > ramp, plateau, progress)


    cal["trend_multiplier"] = 1.0 + cal["trend_strength"] * progress

    cal["trend_multiplier"] = cal["trend_multiplier"].clip(lower=0.0)
    return cal[["Material","ds","trend_multiplier"]]

def compute_trend_table(
    df_cut: pd.DataFrame,
    materials: List[str],
    cutoff: pd.Timestamp,
    periods: int,
    lookback: int = TREND_LOOKBACK,     # keep ( 3monthonly )

    ramp_months: int = TREND_RAMP_MONTHS,
    up_pct: float = TREND_UP_TOTAL_PCT,     # month +threshold

    down_pct: float = TREND_DOWN_TOTAL_PCT, # month -threshold

) -> pd.DataFrame:
    """
    app()in final selected FCST to multiply trend multiplier calendar generate.
    - 3month mixed rulesas strength compute(consecutive/single spike)
    - months linear adjustment, flat
    Return: ["Material","ds","trend_multiplier"]
    """
    strength = _compute_trend_strength_mixed_last3(
        df_cut=df_cut, up_pct=up_pct, down_pct=down_pct, cap=TREND_CAP_ABS
    )
    return _build_trend_multiplier_calendar(
        materials=materials,
        cutoff=cutoff,
        periods=periods,
        strength_map=strength,
        ramp_months=ramp_months,
    )

# =========================

# : SKU forecast (ASHIP baseline) — apply baseonly compute

# =========================

def run_aship_fcst(
    pos_path: str,
    brand: str,
    cutoff: pd.Timestamp = CUTOFF,
    periods: int = PERIODS,
    df_fcst: Optional[pd.DataFrame] = None,     # Snowflakein load

    drop_missing_baseline: bool = False,
    drop_zero_baseline: bool = False,
    strict_three_months: bool = False,
    precomputed: dict | None = None,            # seasonality/bucket/split

    # ▼ USW baseline option

    usw_baseline_lookup: Optional[pd.DataFrame] = None,  # ["Material","USW_baseline"]

    usw_weight: float = 0.40,                              # 0~1 , Default 0.60

) -> Tuple[pd.DataFrame, Dict]:

    # 1) POS → month preprocess

    pos_raw = read_pos_raw(pos_path, header_row=RAW_POS_HEADER_ROW)
    monthly = convert_weekly_to_monthly_long(pos_preprocess(pos_raw, brand=brand), brand=brand, cutoff_for_missing_year=cutoff)
    monthly = add_segment_dummies(monthly)
    monthly["Material"] = _canon_key_series(monthly["Material"])

    # 2) Door@CUTOFF (for debugging)

    door_base, DOOR_COL = build_door_base(monthly, cutoff)

    # 3) ASHIP load/ → baseline(3M avg)

    if df_fcst is None:
        fcst_raw = load_fcst_from_snowflake(brand=brand)  # MATERIAL_KEY, YearMonth, ASHIP

    else:
        fcst_raw = df_fcst[["MATERIAL_KEY","YearMonth","ASHIP"]].copy()
        fcst_raw["MATERIAL_KEY"] = _canon_key_series(fcst_raw["MATERIAL_KEY"])
        fcst_raw["YearMonth"] = pd.to_datetime(fcst_raw["YearMonth"]).dt.to_period("M").dt.to_timestamp(how="start")

    # base_aship = compute_aship_baseline_3m(fcst_raw, cutoff, strict_three_months=strict_three_months)

    base_aship = compute_aship_baseline_4m(fcst_raw, cutoff, strict_three_months=strict_three_months)


    # MATERIAL_KEY → Material mapping

    sel = (monthly.drop_duplicates("Material")[["Material"]].copy())
    sel["MATERIAL_KEY"] = sel["Material"]
    base_aship_lookup = (sel.merge(base_aship, on="MATERIAL_KEY", how="left")[["Material","baseline_units"]]
                           .rename(columns={"baseline_units":"ASHIP_baseline"}))

    # 3.5) USW_baseline

    if usw_baseline_lookup is not None and not usw_baseline_lookup.empty:
        tmp = usw_baseline_lookup.copy()
        tmp["Material"] = _canon_key_series(tmp["Material"])
        tmp["USW_baseline"] = pd.to_numeric(tmp["USW_baseline"], errors="coerce")
        blended = base_aship_lookup.merge(tmp[["Material","USW_baseline"]], on="Material", how="outer")
        w = float(np.clip(usw_weight, 0.0, 1.0))
        def _blend(r):
            a = r.get("ASHIP_baseline")
            u = r.get("USW_baseline")
            if pd.notna(a) and pd.notna(u):
                return w*float(u) + (1.0-w)*float(a)
            if pd.notna(u):
                return float(u)
            if pd.notna(a):
                return float(a)
            return 0.0
        blended["baseline_units"] = blended.apply(_blend, axis=1)
        base_lookup = blended[["Material","baseline_units"]]
        baseline_note = f"blended: USW({w:.2f}) + ASHIP({1.0-w:.2f})"
    else:
        base_lookup = base_aship_lookup.rename(columns={"ASHIP_baseline":"baseline_units"})[["Material","baseline_units"]]
        baseline_note = "ASHIP_3M_avg (no USW blend)"

    # 4) bucket/split/seasonality (precomputed )

    if precomputed is not None:
        df_cut    = precomputed["df_cut"]
        long_list = precomputed["long_list"]
        short_list= precomputed["short_list"]
        seas_long = precomputed["seas_long"]
        seas_seg  = precomputed["seas_seg"]
    else:
        df_cut = monthly.loc[monthly["YearMonth"] <= cutoff].copy()
            # --- ASHIP df_cut (Short/Long ASHIP ) ---

        # fcst_raw: in load/normalize (MATERIAL_KEY, YearMonth, ASHIP)

        aship_ts = fcst_raw.copy()
        # MATERIAL_KEY → Material mapping

        key_map = sel[["Material","MATERIAL_KEY"]]  # sel in only Material

        aship_ts = (aship_ts.merge(key_map, on="MATERIAL_KEY", how="inner")
                            .drop(columns=["MATERIAL_KEY"]))
        # month duplicate sum or mean choose ( sum recommended)

        aship_ts = (aship_ts.groupby(["Material","YearMonth"], as_index=False)
                            .agg(ASHIP=("ASHIP","sum")))

        df_cut = df_cut.merge(aship_ts, on=["Material","YearMonth"], how="left")

        span = (df_cut.dropna(subset=["UPM"]).groupby("Material", as_index=False)["YearMonth"].nunique()
                .rename(columns={"YearMonth":"n_months"}))
        door_at = door_base.rename(columns={DOOR_COL:"door_at_cutoff"})
        sel2 = (span.merge(door_at, on="Material", how="left"))
        sel2["door_at_cutoff"] = pd.to_numeric(sel2["door_at_cutoff"], errors="coerce").fillna(0.0)
        long_list  = sel2.loc[(sel2["n_months"] >= MIN_HISTORY) & (sel2["door_at_cutoff"] > 0), "Material"].tolist()
        short_list = sel2.loc[(sel2["n_months"] <  MIN_HISTORY) & (sel2["door_at_cutoff"] > 0), "Material"].tolist()
        seas_long = build_long_seasonality(df_cut, long_list)
        seas_seg  = build_segment_seasonality(df_cut)

    future_idx = pd.date_range(start=cutoff + pd.offsets.MonthBegin(1), periods=periods, freq="MS")

    # 5) Long — apply base

    if not seas_long.empty:
        calL = (pd.MultiIndex.from_product([sorted(seas_long["Material"].unique()), future_idx],
                                           names=["Material","ds"]).to_frame(index=False))
        calL["Month"] = calL["ds"].dt.month
        calL = (calL.merge(seas_long, on=["Material","Month"], how="left")
                    .merge(base_lookup, on="Material", how="left"))
        if drop_missing_baseline:
            calL = calL.dropna(subset=["baseline_units"])
        calL["baseline_units"] = calL["baseline_units"].fillna(0.0)
        if drop_zero_baseline:
            calL = calL.loc[calL["baseline_units"] > 0]

        calL["baseline_units_adj"] = calL["baseline_units"].astype(float) * BASELINE_BUFFER
        for col_in, col_out in [("factor","yhat_denorm"),
                                ("factor_lower","yhat_lower_denorm"),
                                ("factor_upper","yhat_upper_denorm")]:
            calL[col_out] = calL[col_in].astype(float) * calL["baseline_units_adj"].astype(float)

        calL["method"] = "Long"

        seg_map_all = (
            df_cut[["Material","Segment"]]
            .dropna().drop_duplicates("Material").copy()
        )
        seg_map_all["Segment"] = seg_map_all["Segment"].astype(str).str.strip()
        calL = calL.merge(seg_map_all, on="Material", how="left")   # out_long = calL


        out_long = calL
    else:
        out_long = pd.DataFrame(columns=["Material","ds","Month","factor","factor_lower","factor_upper",
                                         "baseline_units","baseline_units_adj","yhat_denorm",
                                         "yhat_lower_denorm","yhat_upper_denorm","Segment","method"])
        
        

    # 6) Short — apply base

    if short_list:
        calS = (pd.MultiIndex.from_product([sorted(short_list), future_idx],
                                           names=["Material","ds"]).to_frame(index=False))
        calS["Month"] = calS["ds"].dt.month

        if "Segment" in df_cut.columns:
            seg_map = (df_cut.loc[df_cut["Material"].isin(short_list), ["Material","Segment"]]
                             .dropna().drop_duplicates("Material"))
        else:
            seg_dummy_cols = [c for c in df_cut.columns if str(c).startswith("Segment_")]
            tmp = df_cut.loc[df_cut["Material"].isin(short_list), ["Material"] + seg_dummy_cols]
            sums = tmp.groupby("Material")[seg_dummy_cols].sum()
            seg_choice = sums.idxmax(axis=1).to_frame(name="Segment")
            seg_choice["Segment"] = seg_choice["Segment"].str.replace("^Segment_","", regex=True)
            seg_map = seg_choice.reset_index()

        seg_map["Segment"] = seg_map["Segment"].astype(str).str.strip()
        seas_seg["Segment"] = seas_seg["Segment"].astype(str).str.strip()

        calS = (calS.merge(seg_map, on="Material", how="left")
                    .merge(seas_seg[["Segment","Month","factor","factor_lower","factor_upper"]],
                           on=["Segment","Month"], how="left")
                    .merge(base_lookup, on="Material", how="left"))
        # ▼ NEW: Shortin ASHIP seasonality &

        seas_short_aship = build_short_aship_seasonality(df_cut, short_list)
        if seas_short_aship is not None and not seas_short_aship.empty:
            calS = calS.merge(seas_short_aship, on=["Material","Month"], how="left")

            w_usw = float(globals().get("SEAS_SHORT_BLEND_WEIGHT_USW", 0.50))
            # aship_short_factor if present mean, if missing seg factor keep

            calS["factor_blend"] = np.where(
                calS["aship_short_factor"].notna(),
                w_usw * calS["factor"].astype(float) + (1.0 - w_usw) * calS["aship_short_factor"].astype(float),
                calS["factor"].astype(float)
            )
            calS["factor"]        = calS["factor_blend"]
            calS["factor_lower"]  = calS["factor_lower"].astype(float)  # lower bound/upper bound priority segment value keep

            calS["factor_upper"]  = calS["factor_upper"].astype(float)
            calS.drop(columns=["factor_blend"], inplace=True)

        if drop_missing_baseline:
            calS = calS.dropna(subset=["baseline_units"])
        calS["baseline_units"] = calS["baseline_units"].fillna(0.0)
        if drop_zero_baseline:
            calS = calS.loc[calS["baseline_units"] > 0]

        calS["baseline_units_adj"] = calS["baseline_units"].astype(float) * BASELINE_BUFFER
        calS["method"] = "Short"
        calS["yhat_denorm"]       = calS["factor"].astype(float)       * calS["baseline_units_adj"].astype(float)
        calS["yhat_lower_denorm"] = calS["factor_lower"].astype(float) * calS["baseline_units_adj"].astype(float)
        calS["yhat_upper_denorm"] = calS["factor_upper"].astype(float) * calS["baseline_units_adj"].astype(float)
        out_short = calS
    else:
        out_short = pd.DataFrame(columns=["Material","Segment","ds","Month","factor","factor_lower","factor_upper",
                                          "baseline_units","baseline_units_adj","yhat_denorm",
                                          "yhat_lower_denorm","yhat_upper_denorm","method"])

    # 7) combine (trend_multiplier appin apply)

    cols = ["Material","Segment","ds","Month","method","factor","factor_lower","factor_upper",
            "baseline_units","baseline_units_adj","yhat_denorm","yhat_lower_denorm","yhat_upper_denorm"]
    fcst = (pd.concat([out_long[cols], out_short[cols]], axis=0, ignore_index=True)
              .sort_values(["Material","ds"]).reset_index(drop=True))

    debug = {
        "cutoff": str(cutoff.date()),
        "brand": brand,
        "n_all": monthly["Material"].nunique(),
        "n_long": (0 if isinstance(out_long, pd.DataFrame) and out_long.empty else out_long["Material"].nunique()),
        "n_short": (0 if isinstance(out_short, pd.DataFrame) and out_short.empty else out_short["Material"].nunique()),
        "baseline": baseline_note,
        "used_precomputed": precomputed is not None,
        "note": "trend not applied inside M3; use compute_trend_table() at app level",
    }
    return fcst, debug


# =========================

# CLI CLI test example

# =========================

if __name__ == "__main__":
    POS_PATH = "WG.xlsx"
    out, info = run_aship_fcst(POS_PATH, brand="WG", cutoff=CUTOFF, periods=PERIODS,
                               strict_three_months=True, precomputed=None)
    print(info)
    print(out.head(20))
