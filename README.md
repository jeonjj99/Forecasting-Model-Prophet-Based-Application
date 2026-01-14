# Forecasting-Model-Prophet-Based-Application
forecasting model using Prophet algorithm and Streamlit interface


## 0. File roles

### FCST_APP.py
- Streamlit UI and input handling for files, brand, cutoff, periods
- Merges monthly Snowflake data with the monthly POS table built from weekly POS
- Precomputes seasonality and then runs M2 and M3 forecasts so the app can compare outputs

### FCST_M2.py (USW based)
- Builds a baseline from POS derived USW or UPM
- Splits items into Long and Short history
- Uses Long item seasonality when possible and Segment seasonality as fallback for Short items
- Produces monthly forecasts using baseline times seasonality

### FCST_M3.py (ASHIP based)
- Uses ASHIP three month average as the main baseline
- For Long items can blend seasonality signals from USW and ASHIP
- Keeps trend application outside the module and expects the app layer to apply a trend multiplier

---------------------------------------------------------------------------------------------------------------

## 1. Input data

### A. POS raw weekly
- Weekly sellout data from files
- Converted to monthly so everything aligns on a YearMonth key

### B. Optional baseline input for M2
- USW related inputs can be provided to guide baseline construction depending on your setup

### C. Snowflake monthly table
- Monthly ESHIP and ASHIP by material key and YearMonth
- Keys are normalized so Material in the app table matches MATERIAL_KEY from Snowflake

---------------------------------------------------------------------------------------------------------------

## 2. Raw to monthly preprocessing

### 2.1 Why monthly
- Weekly POS is noisy and volatile
- Seasonality and forecast logic runs on a monthly calendar for stability
- A fixed weeks per month factor is used when converting weekly rates to monthly units

### 2.2 Key standardization
- YearMonth is normalized to a consistent monthly key before merges
- Material keys are canonicalized so the join is reliable across sources

---------------------------------------------------------------------------------------------------------------

## 3. Prophet seasonality extraction approach

Common concept
- Fit Prophet with yearly seasonality on a normalized time series
- Extract yearly component
- Convert yearly to a multiplicative factor using 1 plus yearly
- Aggregate to month level
- Normalize so the average factor equals 1
- Optionally map factors into a band range
- Apply special band rules for certain segments such as toe nails

---

## 3.1 Long item seasonality in M3 (USW and ASHIP blending)

Step 1. USW based factor
- Build a monthly series from POS metrics such as UPM
- Normalize within each year using min max scaling
- Fit Prophet with yearly seasonality and multiplicative mode
- Extract yearly component and build a monthly factor
- Normalize to mean 1

Step 2. ASHIP based factor
- If ASHIP history is sufficient, repeat the same process on ASHIP
- Normalize within year
- Fit Prophet yearly only
- Build monthly factor and normalize to mean 1

Step 3. Blend factors
- Blend the two normalized factors using a weight for USW
- If ASHIP factor is not available, use USW only

Step 4. Band mapping and segment exceptions
- Map the blended factor into a defined range
- Use a stronger range for toe nails type segments

Outputs
- Material
- Month
- factor
- factor_lower
- factor_upper

---

## 3.2 Segment seasonality for Short items and fallback

Goal
- Create one monthly factor per segment
- Short items use their segment factor when item level seasonality is unreliable

Process
- Determine Segment label from available fields if needed
- Build segment level monthly series and normalize within year
- If enough months, fit Prophet yearly only
- If not enough months, fall back to month averages
- Normalize to mean 1 and map into the band range
- Apply toe nails exception range if applicable

---

## 3.3 Note on M2 differences
- M2 can also blend seasonality signals
- In M2 the ASHIP seasonality piece may be built from month averages after normalization rather than always refitting Prophet on ASHIP
- The final idea is the same: baseline times seasonality

---------------------------------------------------------------------------------------------------------------

## 4. Forecast construction (baseline times seasonality)

### 4.1 Long and Short split
- Compute months of history per material
- Long items have at least a minimum number of months
- Short items have less than that threshold

### 4.2 Future calendar
- Create future YearMonth values starting after cutoff for the chosen number of periods

### 4.3 Forecast formula
- baseline_units is the baseline demand level
- factor is the monthly seasonality multiplier
- yhat_denorm equals factor times adjusted baseline_units

### 4.4 Baseline in M3
- Default baseline is ASHIP three month average
- Optional blending with a USW based baseline can be used

### 4.5 Trend application location
- Trend is not applied inside M3
- The app layer computes a trend table and applies a multiplier after baseline times seasonality

---------------------------------------------------------------------------------------------------------------

## 5. Streamlit app implementation flow (FCST_APP.py)

### 5.1 Merge Snowflake ASHIP into the app monthly table
- Copy and normalize keys on df_fcst_all
- Build a key map between df_cut Material and Snowflake MATERIAL_KEY
- Extract monthly ASHIP time series and merge into df_cut by Material and YearMonth

### 5.2 Precompute seasonality
- Build long item seasonality once
- Build segment seasonality once
- Store in a precomputed object and pass into M2 and M3 so modules do not recompute

### 5.3 Run M2 then standardize output names
- Run run_usw_fcst with precomputed seasonality
- Rename key columns for the UI, for example
  - yhat_denorm to USW_FCST
  - factor to USW_factor
  - baseline_units to USW_baseline

### 5.4 Run M3
- Run run_aship_fcst similarly and bring results into the app for comparison and review

---------------------------------------------------------------------------------------------------------------

## 6. Key output columns

- factor: monthly seasonality multiplier after band mapping
- baseline_units: baseline demand level such as ASHIP three month average
- yhat_denorm: forecast before applying any app level trend multiplier

---------------------------------------------------------------------------------------------------------------

## One line summary
- Convert weekly POS to monthly
- Extract monthly seasonality with Prophet yearly component
- Produce forecasts using baseline times seasonality
- Apply trend in the Streamlit app layer and present M2 and M3 outputs side by side
