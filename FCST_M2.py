# ============================================================
# POS (Weekly→Monthly) + USW → 6-month FCST for all SKUs (Long/Short auto)
# - No Snowflake/Group usage
# - Built-in brand-specific Week→Month mapping/preprocessing
# - [Change] Trend adjustment is NOT applied inside M2
# · This module always outputs a "pre-trend" base FCST only
# · After the final selection (USW/ASHIP) in the central app, apply the multiplier from compute_trend_table() once
# ============================================================

from __future__ import annotations

import re
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from prophet import Prophet

pd.options.mode.copy_on_write = True

# =========================
# Base parameters
# =========================
WEEKS_PM   = 4.3
CUTOFF     = pd.Timestamp("2025-08-01")  # Cutoff month (month start)
PERIODS    = 6
MIN_HISTORY = 24
BAND_LOW, BAND_HIGH = 0.851425, 1.148575
NORMALIZE_MEAN_TO_ONE = True
AGG_METHOD_SEG = "mean"      # 'mean'|'median'
MIN_MONTHS_FOR_PROPHET_SEG = 10
RAW_POS_HEADER_ROW = 1
BASELINE_BUFFER = 1

# =========================
# (Reference) Trend parameters — NOT applied internally / used only in utilities
# =========================
TREND_LOOKBACK = 3
TREND_RAMP_MONTHS = 4
TREND_UP_TOTAL_PCT = 0.20
TREND_DOWN_TOTAL_PCT = 0.20
TREND_CAP_ABS = 0.50
TREND_EPS = 1e-8

# ===== Seasonality blend options (Long-only) =====
SEAS_BLEND_LONG_WITH_ASHIP = True   # For Long items only: USW×w + ASHIP×(1-w)
SEAS_BLEND_WEIGHT_USW      = 0.7    # Final factor = w*USW + (1-w)*ASHIP
SEAS_LONG_MIN_MONTHS       = 24     # Long threshold (observed months)
SEAS_ASHIP_MIN_HISTORY     = 6      # Minimum valid months to build ASHIP seasonality

# =========================
# Week→Month mapping (by brand)
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

# ===== Additional common utilities =====
import unicodedata
import re as _re

SEGMENT_CANON_MAP = {
    "impress": "imPRESS",
    "preglued nails": "PreGlued Nails",
    "french nails": "French Nails",
    "decorated nails": "Decorated Nails",
    "color nails": "Color Nails",
    "toe nails": "Toe Nails",  # Standardize Toe Nails as well if it appears
    "impress toe nail": "Toe Nails",
}

def canon_seg_series(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .apply(lambda x: unicodedata.normalize("NFKC", x))   # Unicode normalization
              .str.replace(r"\s+", " ", regex=True)                # Multiple internal spaces → single space
              .str.strip()
              .str.lower())

def normalize_segment(s: pd.Series) -> pd.Series:
    # Map to standardized name using canonical key (lowercase)
    key = canon_seg_series(s)
    mapped = key.replace(SEGMENT_CANON_MAP)
    return mapped

def _norm_minmax_yearwise(s: pd.Series) -> pd.Series:
    if s.isna().all():
        return pd.Series(np.nan, index=s.index)
    vmin, vmax = s.min(), s.max()
    if pd.isna(vmin) or pd.isna(vmax) or np.isclose(vmin, vmax):
        return pd.Series(0.5, index=s.index, dtype=float)
    return (s - vmin) / (vmax - vmin)

def _aship_monthly_seasonality(df: pd.DataFrame,
                               key_cols: list[str],
                               min_hist_months: int = 6) -> pd.DataFrame:
    """
    Input: df requires ['YearMonth','ASHIP']; YearMonth should be a month-start timestamp
    Output: monthly (1..12) factor per key ('factor_aship') — scaled to mean=1
    """
    work = df.copy()
    work["YEAR"] = work["YearMonth"].dt.year
    work["Mo"]   = work["YearMonth"].dt.month

    work["ASHIP_norm"] = work.groupby(key_cols + ["YEAR"], dropna=False)["ASHIP"] \
                             .transform(_norm_minmax_yearwise)

    valid = work.dropna(subset=["ASHIP_norm"])
    counts = (valid.groupby(key_cols, dropna=False)["ASHIP_norm"]
                    .count().reset_index(name="n"))
    ok_keys = set(counts.loc[counts["n"] >= min_hist_months][key_cols]
                        .itertuples(index=False, name=None))
    if not ok_keys:
        return pd.DataFrame(columns=key_cols + ["Mo","factor_aship"])

    mo_avg = (valid.groupby(key_cols + ["Mo"], as_index=False, dropna=False)["ASHIP_norm"]
                    .mean().rename(columns={"ASHIP_norm":"factor_aship"}))
    mo_avg["mean_key"] = mo_avg.groupby(key_cols, dropna=False)["factor_aship"].transform("mean")
    mo_avg["factor_aship"] = mo_avg["factor_aship"] / mo_avg["mean_key"]
    mo_avg = mo_avg.drop(columns=["mean_key"])

    mask = mo_avg[key_cols].apply(tuple, axis=1).isin(ok_keys)
    return mo_avg.loc[mask].reset_index(drop=True)

# =========================
# POS preprocessing (weekly→monthly)
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
        # Convert to standardized display name (Title Case)
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
# USW-based baseline
# =========================
def normalize_usw_df(df_usw_raw: pd.DataFrame) -> pd.DataFrame:

    cols = {
        re.sub(r"[^a-z0-9 ]", "", str(c).strip().lower()): c
        for c in df_usw_raw.columns
    }

    mat_aliases = ["material","sku","item","materialcode","material_code","materialid","material id","materialno","material number"]
    mat_col = next((cols[a] for a in mat_aliases if a in cols), None)
    if mat_col is None:
        raise ValueError("USW 파일에서 Material/SKU 컬럼을 찾지 못했습니다.")
    usw_col = None
    for k, orig in cols.items():
        if k in ("usw","u s w","u_s_w","u-s-w"):
            usw_col = orig; break
    if usw_col is None:
        cand = [c for c in df_usw_raw.columns if c != mat_col and pd.api.types.is_numeric_dtype(df_usw_raw[c])]
        if not cand: raise ValueError("U/S/W 숫자형 컬럼을 찾지 못했습니다.")
        usw_col = cand[0]
    df = df_usw_raw[[mat_col, usw_col]].copy()
    df.columns = ["Material","USW"]
    df["Material"] = df["Material"].astype(str).str.strip()
    df["USW"] = pd.to_numeric(df["USW"], errors="coerce").fillna(0.0)
    return df.groupby("Material", as_index=False)["USW"].mean()

def build_door_base(df_all: pd.DataFrame, cutoff: pd.Timestamp) -> Tuple[pd.DataFrame, str]:
    cutoff_ts = pd.Timestamp(cutoff).to_period("M").to_timestamp(how = "start")
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

def compute_baseline(usw_df: pd.DataFrame, door_base: pd.DataFrame, door_col: str) -> pd.DataFrame:
    base = (usw_df[["Material","USW"]].drop_duplicates("Material")
              .merge(door_base[["Material", door_col]], on="Material", how="left"))
    base[door_col] = pd.to_numeric(base[door_col], errors="coerce")
    base["USW"] = pd.to_numeric(base["USW"], errors="coerce").fillna(0.0)

    base["USW_month"] = base["USW"] * float(WEEKS_PM)  # Weekly→Monthly conversion
    base["baseline_units"] = base["USW_month"] * base[door_col].fillna(0.0)

    out = base[["Material", "USW", "USW_month", door_col, "baseline_units"]].copy()
    out = out.rename(columns={door_col: "Door_at_cutoff"})
    return out

# =========================
# Seasonality (UPM-based)
# =========================
def build_long_seasonality(df_hist_cut: pd.DataFrame, eligible_mats: list[str]) -> pd.DataFrame:
    """
    For Long items (with sufficient monthly history),
    blend USW-based seasonality (existing) and ASHIP-based seasonality
    (per-year/per-item min-max → monthly average → mean=1) to produce the final monthly seasonality vector.
    For Short items (short history), use the existing USW seasonality only.

    Required columns:
      - df_hist_cut: ['Material','Segment','YearMonth','UPM', (optional) 'ASHIP']
      - YearMonth: ideally normalized to a month-start timestamp
    """

    # ---- Load global/default parameters (use defaults if missing) ----
    BLEND_ENABLED       = bool(globals().get("SEAS_BLEND_LONG_WITH_ASHIP", True))
    WEIGHT_USW          = float(globals().get("SEAS_BLEND_WEIGHT_USW", 0.7))   # Final = w*USW + (1-w)*ASHIP
    LONG_MIN_MONTHS     = int(globals().get("SEAS_LONG_MIN_MONTHS", 24))       # Long threshold
    ASHIP_MIN_HISTORY   = int(globals().get("SEAS_ASHIP_MIN_HISTORY", 6))      # Minimum months for ASHIP
    # Uses existing global constants/functions: MIN_HISTORY, BAND_LOW/HIGH, NORMALIZE_MEAN_TO_ONE, _minmax_per_year, _linmap, Prophet

    # ⬇️ Material → seg_key (lowercase) mapping (used for Toe Nails range exception)
    _seg_map = (
        df_hist_cut[["Material","Segment"]]
        .dropna()
        .drop_duplicates("Material")
        .copy()
    )
    _seg_map["seg_key"] = _seg_map["Segment"].astype(str).str.strip().str.lower()
    _seg_lookup = dict(zip(_seg_map["Material"], _seg_map["seg_key"]))

    rows = []
    # Process only eligible_mats (target SKUs)
    for mat, g in df_hist_cut[df_hist_cut["Material"].isin(eligible_mats)].groupby("Material", sort=False):
        g = g.sort_values("YearMonth").copy()
        g["ds"] = g["YearMonth"].values
        g["YEAR"] = g["ds"].dt.year
        g["Month"] = g["ds"].dt.month

        # ---------- USW-based seasonality (keep original logic) ----------
        g["UPM"] = pd.to_numeric(g["UPM"], errors="coerce")
        g["UPM_norm"] = g.groupby("YEAR")["UPM"].transform(_minmax_per_year)

        train = g[["ds","UPM_norm"]].rename(columns={"UPM_norm":"y"}).dropna()
        # Skip if below minimum training months
        if train["ds"].nunique() < MIN_HISTORY:
            continue

        m = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            seasonality_mode="multiplicative", interval_width=0.95
        )
        m.fit(train)
        comp = m.predict(train[["ds"]])[["ds","yearly"]].copy()
        comp["factor_raw"] = 1.0 + comp["yearly"]
        comp["Month"] = comp["ds"].dt.month
        month_avg = comp.groupby("Month", as_index=False)["factor_raw"].mean().sort_values("Month")

        # Mean=1 normalization (original logic)
        if NORMALIZE_MEAN_TO_ONE:
            mu = month_avg["factor_raw"].mean()
            month_avg["usw_factor_norm"] = (
                month_avg["factor_raw"] / mu if (pd.notna(mu) and not np.isclose(mu, 0.0)) else month_avg["factor_raw"]
            )
        else:
            month_avg["usw_factor_norm"] = month_avg["factor_raw"]

        # ---------- ASHIP-based seasonality (Long only, only when sufficient data) ----------
        # Default: do not use ASHIP seasonality
        month_avg["aship_factor_norm"] = np.nan

        # Determine Long status (based on history months)
        n_months_hist = int(g["ds"].nunique())
        is_long = (n_months_hist >= LONG_MIN_MONTHS)

        if BLEND_ENABLED and is_long and ("ASHIP" in g.columns):
            # Available ASHIP months
            g["ASHIP"] = pd.to_numeric(g["ASHIP"], errors="coerce")
            aship_valid_months = int(g["ASHIP"].dropna().shape[0])

            if aship_valid_months >= ASHIP_MIN_HISTORY:
                # Year/product min-max normalization
                g["ASHIP_norm"] = g.groupby("YEAR")["ASHIP"].transform(_minmax_per_year)
                # Monthly average
                a_month = (
                    g.dropna(subset=["ASHIP_norm"])
                     .groupby("Month", as_index=False)["ASHIP_norm"]
                     .mean()
                     .rename(columns={"ASHIP_norm":"aship_factor_raw"})
                )
                # Mean=1 normalization
                if not a_month.empty:
                    mu_a = a_month["aship_factor_raw"].mean()
                    a_month["aship_factor_norm"] = (
                        a_month["aship_factor_raw"] / mu_a
                        if (pd.notna(mu_a) and not np.isclose(mu_a, 0.0))
                        else a_month["aship_factor_raw"]
                    )
                    # Merge into month_avg
                    month_avg = month_avg.merge(
                        a_month[["Month","aship_factor_norm"]],
                        on="Month", how="left"
                    )

        # ---------- Monthly blending: blended_norm = w*USW + (1-w)*ASHIP ----------
        # If ASHIP is missing or conditions not met, use USW only
        if "aship_factor_norm" in month_avg.columns:
            usw = month_avg["usw_factor_norm"].astype(float)
            ash = month_avg["aship_factor_norm"].astype(float)
            month_avg["blended_norm"] = np.where(ash.notna(), WEIGHT_USW*usw + (1.0 - WEIGHT_USW)*ash, usw)
        else:
            month_avg["blended_norm"] = month_avg["usw_factor_norm"].astype(float)

        # ---------- Range mapping (including Toe Nails exception) ----------
        seg_key = _seg_lookup.get(mat, "")
        if seg_key == "toe nails":
            low, high = 0.6, 1.4
        else:
            low, high = BAND_LOW, BAND_HIGH

        mapped = _linmap(month_avg["blended_norm"].to_numpy(float), low, high)

        # ---------- Output ----------
        out = month_avg[["Month"]].copy()
        out.insert(0, "Material", mat)
        out["factor"] = mapped
        out["factor_lower"] = out["factor"]
        out["factor_upper"] = out["factor"]
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["Material","Month","factor","factor_lower","factor_upper"])
    return pd.concat(rows, ignore_index=True)



def build_segment_seasonality(df_all_cut: pd.DataFrame) -> pd.DataFrame:
    work = df_all_cut.copy()

    # Segment fix: if Segment column is missing, use dummies → recover most frequent Segment
    if "Segment" not in work.columns:
        seg_dummy_cols = [c for c in work.columns if str(c).startswith("Segment_")]
        if not seg_dummy_cols:
            raise ValueError("Segment 컬럼 또는 Segment_* 더미가 필요합니다.")
        sums = work.groupby("Material")[seg_dummy_cols].sum()
        seg_choice = sums.idxmax(axis=1).to_frame(name="Segment")
        seg_choice["Segment"] = seg_choice["Segment"].str.replace("^Segment_", "", regex=True)
        work = work.merge(seg_choice.reset_index(), on="Material", how="left")

    # Build segment time series using within-year min-max normalized UPM
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
            # Prophet path: yearly component → factor_raw = 1 + yearly
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
            # Fallback: monthly mean y → mean=1 normalize → build comp via month mapping
            tmp = g.copy()
            tmp["Month"] = tmp["ds"].dt.month
            mo = tmp.groupby("Month")["y"].mean()
            # Mean=1 normalization (avoid zero denominator)
            mo = mo / (mo.mean() if pd.notna(mo.mean()) and not np.isclose(mo.mean(), 0) else 1.0)
            comp = tmp[["ds","Month"]].copy()
            comp["factor_raw"] = comp["Month"].map(mo).astype(float)

        # Monthly average → linear mapping (Toe Nails exception)
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
            low, high = 0.7, 1.3
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


# =========================
# Trend calendar utility (used to apply final multiplication in app)
# =========================
def _compute_trend_strength_per_material(df_cut: pd.DataFrame, k: int) -> pd.DataFrame:
    work = df_cut.sort_values(["Material","YearMonth"]).copy()
    last_k = (work.groupby("Material")
                    .tail(k)[["Material","YearMonth","UPM"]]
                    .dropna(subset=["UPM"]))
    last_k["idx"] = last_k.groupby("Material").cumcount()

    strengths = []
    for mat, g in last_k.groupby("Material", sort=False):
        if g["UPM"].notna().sum() < 2:
            strengths.append((mat, 0.0)); continue
        x = g["idx"].to_numpy(float)
        y = g["UPM"].to_numpy(float)
        slope = np.polyfit(x, y, 1)[0]   # UPM change per month
        mu = np.nanmean(y)
        pct_per_month = slope / max(mu, TREND_EPS)   # Monthly % slope

        th_up_pm   = TREND_UP_TOTAL_PCT / max(k-1, 1)
        th_down_pm = TREND_DOWN_TOTAL_PCT / max(k-1, 1)

        if   pct_per_month >=  th_up_pm:
            raw =  pct_per_month / th_up_pm
        elif pct_per_month <= -th_down_pm:
            raw =  pct_per_month / (-th_down_pm)
        else:
            raw = 0.0

        strength = float(np.clip(raw, -TREND_CAP_ABS, TREND_CAP_ABS))
        strengths.append((mat, strength))

    return pd.DataFrame(strengths, columns=["Material","trend_strength"])

def _build_trend_multiplier_calendar(materials: List[str],
                                     cutoff: pd.Timestamp,
                                     periods: int,
                                     strength_map: pd.DataFrame,
                                     ramp_months: int) -> pd.DataFrame:
    future_idx = pd.date_range(start=cutoff + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    cal = pd.MultiIndex.from_product([sorted(materials), future_idx], names=["Material","ds"]).to_frame(index=False)
    cal["h"] = cal.groupby("Material")["ds"].cumcount() + 1

    cal = cal.merge(strength_map, on="Material", how="left")
    cal["trend_strength"] = cal["trend_strength"].fillna(0.0)

    ramp = max(1, int(ramp_months))
    ramp_frac = np.minimum(cal["h"] / ramp, 1.0)
    cal["trend_multiplier"] = 1.0 + cal["trend_strength"] * ramp_frac
    cal["trend_multiplier"] = cal["trend_multiplier"].clip(lower=0.0)
    return cal[["Material","ds","trend_multiplier"]]

def compute_trend_table(
    df_cut: pd.DataFrame,
    materials: List[str],
    cutoff: pd.Timestamp,
    periods: int,
    lookback: int = TREND_LOOKBACK,
    ramp_months: int = TREND_RAMP_MONTHS,
) -> pd.DataFrame:
    strength = _compute_trend_strength_per_material(df_cut, k=lookback)
    return _build_trend_multiplier_calendar(
        materials=materials,
        cutoff=cutoff,
        periods=periods,
        strength_map=strength,
        ramp_months=ramp_months,
    )

# =========================
# Main: forecast all SKUs (ignore Group) — output base only (no trend applied)
# =========================
def run_usw_fcst(
    pos_path: str,
    usw_path: str,
    brand: str,
    cutoff: pd.Timestamp = CUTOFF,
    periods: int = PERIODS,
    drop_missing_baseline: bool = False,
    drop_zero_baseline: bool = False,
    precomputed: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    # 1) POS → monthly preprocessing
    pos_raw = read_pos_raw(pos_path, header_row=RAW_POS_HEADER_ROW)

    monthly = convert_weekly_to_monthly_long(
        pos_preprocess(pos_raw, brand=brand),   # ← Brand-specific preprocessing (TG: STORE filter, etc.)
        brand=brand,                            # ← Brand-specific Week→Month mapping
        cutoff_for_missing_year=cutoff
    )

    monthly = add_segment_dummies(monthly)

    # 2) baseline (USW × Door@CUTOFF)
    usw_df   = normalize_usw_df(pd.read_excel(usw_path))
    door_base, DOOR_COL = build_door_base(monthly, cutoff)
    baseline = compute_baseline(usw_df, door_base, DOOR_COL)

    # 3) df_cut / Long-Short / Seasonality (prefer precomputed)
    if precomputed is not None:
        df_cut    = precomputed["df_cut"]
        long_list = precomputed["long_list"]
        short_list= precomputed["short_list"]
        seas_long = precomputed["seas_long"]
        seas_seg  = precomputed["seas_seg"]
    else:
        df_cut = monthly.loc[monthly["YearMonth"] <= cutoff].copy()

        span = (df_cut.dropna(subset=["UPM"])
                      .groupby("Material", as_index=False)["YearMonth"]
                      .nunique()
                      .rename(columns={"YearMonth":"n_months"}))
        door_at = door_base.rename(columns={DOOR_COL:"door_at_cutoff"})
        sel = span.merge(door_at, on="Material", how="left")
        sel["door_at_cutoff"] = pd.to_numeric(sel["door_at_cutoff"], errors="coerce").fillna(0.0)

        long_list  = sel.loc[(sel["n_months"] >= MIN_HISTORY) & (sel["door_at_cutoff"] > 0), "Material"].tolist()
        short_list = sel.loc[(sel["n_months"] <  MIN_HISTORY) & (sel["door_at_cutoff"] > 0), "Material"].tolist()

        seas_long = build_long_seasonality(df_cut, long_list)
        seas_seg  = build_segment_seasonality(df_cut)

        # ▼ Shared for Long/Short: Material→Segment mapping table (recover from dummies if Segment missing)
    if "Segment" in df_cut.columns and df_cut["Segment"].notna().any():
        seg_map_all = (
            df_cut[["Material","Segment"]]
            .dropna()
            .drop_duplicates("Material")
            .copy()
        )
    else:
        seg_dummy_cols = [c for c in df_cut.columns if str(c).startswith("Segment_")]
        if seg_dummy_cols:
            sums = df_cut.groupby("Material")[seg_dummy_cols].sum()
            seg_choice = sums.idxmax(axis=1).to_frame(name="Segment")
            seg_choice["Segment"] = seg_choice["Segment"].str.replace("^Segment_", "", regex=True)
            seg_map_all = seg_choice.reset_index()
        else:
            seg_map_all = pd.DataFrame(columns=["Material","Segment"])

    seg_map_all["Segment"] = seg_map_all["Segment"].astype(str).str.strip()


    # 4) Forecast calendar (future month index)
    future_idx = pd.date_range(start=cutoff + pd.offsets.MonthBegin(1), periods=periods, freq="MS")

        # ▼ Right after creating baseline, df_cut, etc. at the start of run_usw_fcst() (before Long/Short split)
    empty_cols = ["Material","Segment","ds","Month","factor","factor_lower","factor_upper",
                "baseline_units","baseline_units_adj","yhat_denorm","yhat_lower_denorm","yhat_upper_denorm",
                "USW","USW_month","Door_at_cutoff","method"]
    out_long  = pd.DataFrame(columns=empty_cols)
    out_short = pd.DataFrame(columns=empty_cols)


    # 5) Long — no trend applied
    # 5) Long — no trend applied
    if not seas_long.empty:
        calL = (
            pd.MultiIndex.from_product([sorted(seas_long["Material"].unique()), future_idx],
                                    names=["Material","ds"])
            .to_frame(index=False)
        )
        calL["Month"] = calL["ds"].dt.month
        calL = (calL
                .merge(seas_long, on=["Material","Month"], how="left")
                .merge(baseline,  on="Material",           how="left"))

        if drop_missing_baseline:
            calL = calL.dropna(subset=["baseline_units"])
        calL["baseline_units"] = calL["baseline_units"].fillna(0.0)
        if drop_zero_baseline:
            calL = calL.loc[calL["baseline_units"] > 0]

        calL["baseline_units_adj"] = calL["baseline_units"].astype(float) * BASELINE_BUFFER

        for col_in, col_out in [
            ("factor",       "yhat_denorm"),
            ("factor_lower", "yhat_lower_denorm"),
            ("factor_upper", "yhat_upper_denorm"),
        ]:
            calL[col_out] = calL[col_in].astype(float) * calL["baseline_units_adj"].astype(float)

        # ▼ Always attach Segment for Long as well (do NOT create placeholders!)
        calL = calL.merge(seg_map_all, on="Material", how="left")

        calL["method"] = "Long"
        out_long = calL
    else:
        out_long = pd.DataFrame(columns=[
            "Material","Segment","ds","Month","factor","factor_lower","factor_upper",
            "baseline_units","baseline_units_adj","yhat_denorm","yhat_lower_denorm","yhat_upper_denorm",
            "USW","USW_month","Door_at_cutoff","method"
        ])



    # 6) Short — no trend applied
    if short_list:
        calS = (pd.MultiIndex.from_product([sorted(short_list), future_idx], names=["Material","ds"]).to_frame(index=False))
        calS["Month"] = calS["ds"].dt.month

        if "Segment" in df_cut.columns:
            seg_map = (df_cut.loc[df_cut["Material"].isin(short_list), ["Material","Segment"]]
                             .dropna().drop_duplicates("Material"))
        else:
            seg_dummy_cols = [c for c in df_cut.columns if str(c).startswith("Segment_")]
            tmp  = df_cut.loc[df_cut["Material"].isin(short_list), ["Material"] + seg_dummy_cols]
            sums = tmp.groupby("Material")[seg_dummy_cols].sum()
            seg_choice = sums.idxmax(axis=1).to_frame(name="Segment")
            seg_choice["Segment"] = seg_choice["Segment"].str.replace("^Segment_","", regex=True)
            seg_map = seg_choice.reset_index()

        seg_map["Segment"]  = seg_map["Segment"].astype(str).str.strip()
        seas_seg["Segment"] = seas_seg["Segment"].astype(str).str.strip()

        calS = (calS.merge(seg_map, on="Material", how="left")
                    .merge(seas_seg[["Segment","Month","factor","factor_lower","factor_upper"]],
                           on=["Segment","Month"], how="left")
                    .merge(baseline, on="Material", how="left"))

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
                                          "baseline_units","yhat_denorm","yhat_lower_denorm","yhat_upper_denorm",
                                          "method","baseline_units_adj","USW","USW_month","Door_at_cutoff"])

    # 7) Combine — ★ keep debug columns (USW, USW_month, Door_at_cutoff)
    cols = ["Material","Segment","ds","Month","method","factor","factor_lower","factor_upper",
            "baseline_units","baseline_units_adj","yhat_denorm","yhat_lower_denorm","yhat_upper_denorm",
            "USW","USW_month","Door_at_cutoff"]
    # Some columns may be missing; take safe intersection
    cols = [c for c in cols if c in out_long.columns or c in out_short.columns]
    fcst = (pd.concat([out_long[cols], out_short[cols]],
                      axis=0, ignore_index=True)
              .sort_values(["Material","ds"]).reset_index(drop=True))

    debug = {
        "cutoff": str(cutoff.date()),
        "n_all": monthly["Material"].nunique(),
        "n_long": (0 if isinstance(out_long, pd.DataFrame) and out_long.empty else out_long["Material"].nunique()),
        "n_short": (0 if isinstance(out_short, pd.DataFrame) and out_short.empty else out_short["Material"].nunique()),
        "brand": brand,
        "note": "trend not applied inside M2; use compute_trend_table() at app level",
    }
    return fcst, debug

if __name__ == "__main__":
    POS_PATH = r"Z:\CSS\POS Data\Walgreens 2.0 (w. WAG Unify+) v2.xlsx"
    USW_PATH = r"WG_Ranking.xlsx"
    out, info = run_usw_fcst(POS_PATH, USW_PATH, brand="WG", cutoff=CUTOFF, periods=PERIODS)
    print(info)
    print(out.head(20))