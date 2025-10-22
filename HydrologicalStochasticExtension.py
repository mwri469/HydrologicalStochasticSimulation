"""
stochastic_extend_vcsn.py

Create a single stochastic extension for VCSN virtual stations by sampling monthly.
This script:
 - reads observed long gauges (Hunua, Waitakere) DAILY rainfall 1853-2025
 - reads VCSN virtual station DAILY rainfall (1972-2025)
 - aggregates both to MONTHLY totals
 - fits month-by-month models:
     * bulk: Gamma distribution fitted to non-extreme months
     * tail: GPD fitted to exceedances above a monthly threshold (95th percentile)
     * occurrence: logistic regression predicting virtual station exceedance from gauge exceedances
 - simulates one extended MONTHLY record for months where VCSN is missing (1853-1971)
 - disaggregates back to DAILY using proportional scaling from VCSN climatology
 - saves the combined 1853-2025 CSV for each virtual station

Requirements:
 pandas, numpy, scipy, statsmodels, tqdm (optional, for progress bar)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from tqdm import tqdm

# ---------- User config ----------
DATA_DIR = Path(r"P:\1011941\1011941.3000\WorkingMaterial\01 IPCC6 MODELLING\01 COLLATE DOWNSCALED NIWA DATA\02 OUTPUT\ConvertedVCSNdata")
GAUGE_DIR = Path(r"P:\1011941\1011941.3000\WorkingMaterial\01 IPCC6 MODELLING\01 COLLATE DOWNSCALED NIWA DATA\00 DATA")
OUTPUT_DIR = Path("../../../02 OUTPUT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "hunua_vcsn": DATA_DIR / "HunuaRanges_pr_29738.csv",
    "waitakere_vcsn": DATA_DIR / "WaitakereRanges_pr_25923.csv",
    "onehunga_vcsn": DATA_DIR / "Onehunga_pr_27592.csv",
    "upper_waikato_vcsn": DATA_DIR / "UpperWaikato_pr_29940.csv",
    "lower_waikato_vcsn": DATA_DIR / "LowerWaikato_pr_28200.csv",
    "waipariver_vcsn": DATA_DIR / "WaipaRiver_pr_29278.csv",
    "hunua_record": GAUGE_DIR / "PD00_HunuaRanges_RF_1853-2025.csv",
    "waitakere_record": GAUGE_DIR / "PD00_WaitakereRanges_RF_1853-2025.csv"
}

VIRTUAL_STATIONS = [
    ("Onehunga", FILES["onehunga_vcsn"]),
    ("UpperWaikato", FILES["upper_waikato_vcsn"]),
    ("LowerWaikato", FILES["lower_waikato_vcsn"]),
    ("WaipaRiver", FILES["waipariver_vcsn"])
]

GAUGES = {
    "Hunua": FILES["hunua_record"],
    "Waitakere": FILES["waitakere_record"]
}

THRESH_QUANTILE = 0.95
MIN_EXCEEDANCES_GPD = 4
MIN_BULK_SAMPLES_GAMMA = 10

# ---------- Helper functions ----------
def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent date and rainfall columns with daily timesteps."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    rain_col = df.columns[-1]
    df["Rainfall"] = df[rain_col].astype(float)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df[["Date", "Year", "Month", "Rainfall"]]

def read_and_prepare_daily(path: Path) -> pd.DataFrame:
    """Read daily rainfall CSV and process timestamps."""
    df = pd.read_csv(path)
    df = process_datetime(df)
    return df

def to_monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to monthly rainfall totals."""
    monthly = df.groupby(["Year", "Month"], as_index=False)["Rainfall"].sum()
    monthly["YearMonth"] = pd.to_datetime(monthly[["Year", "Month"]].assign(day=1))
    return monthly[["YearMonth", "Year", "Month", "Rainfall"]]

def month_grouped_percentile(series: pd.Series, months: pd.Series, q: float):
    """Compute month-by-month quantile."""
    result = {}
    for m in range(1, 13):
        vals = series[months == m].dropna().values
        result[m] = np.quantile(vals, q) if len(vals) > 0 else np.nan
    return result

# ---------- Fit monthly models ----------
def fit_monthly_models(vcsn_monthly: pd.DataFrame, gauges_monthly: pd.DataFrame, station_name: str):
    """
    Fit month-specific models for MONTHLY rainfall totals.
    Returns models dict with gamma, GPD, and logistic regression parameters.
    """
    models = {}
    
    # Merge VCSN with gauges on YearMonth for conditional modeling
    merged = pd.merge(
        vcsn_monthly, 
        gauges_monthly[["YearMonth", "Hunua_Rainfall", "Waitakere_Rainfall"]],
        on="YearMonth",
        how="inner"
    )
    
    for m in range(1, 13):
        month_data = merged[merged["Month"] == m].copy()
        vcsn_vals = month_data["Rainfall"].dropna()
        
        if len(vcsn_vals) < MIN_BULK_SAMPLES_GAMMA:
            print(f"  Month {m}: Insufficient data ({len(vcsn_vals)} samples)")
            continue
        
        # Fit bulk (non-extreme) with Gamma
        threshold = np.quantile(vcsn_vals, THRESH_QUANTILE)
        bulk = vcsn_vals[vcsn_vals <= threshold]
        
        if len(bulk[bulk > 0]) >= MIN_BULK_SAMPLES_GAMMA:
            shape, loc, scale = stats.gamma.fit(bulk[bulk > 0], floc=0)
        else:
            shape, scale = np.nan, np.nan
        
        # Fit tail with GPD
        exceedances = vcsn_vals[vcsn_vals > threshold] - threshold
        if len(exceedances) >= MIN_EXCEEDANCES_GPD:
            c, loc_gpd, scale_gpd = stats.genpareto.fit(exceedances, floc=0)
        else:
            c, scale_gpd = np.nan, np.nan
        
        # Fit logistic regression for exceedance occurrence
        month_data["VCSN_exceed"] = (month_data["Rainfall"] > threshold).astype(int)
        month_data["Hunua_exceed"] = (month_data["Hunua_Rainfall"] > 
                                       np.quantile(month_data["Hunua_Rainfall"].dropna(), THRESH_QUANTILE)).astype(int)
        month_data["Waitakere_exceed"] = (month_data["Waitakere_Rainfall"] > 
                                           np.quantile(month_data["Waitakere_Rainfall"].dropna(), THRESH_QUANTILE)).astype(int)
        
        # Prepare logistic regression
        X = month_data[["Hunua_exceed", "Waitakere_exceed"]].values
        X = sm.add_constant(X)
        y = month_data["VCSN_exceed"].values
        
        try:
            logit_model = sm.Logit(y, X).fit(disp=0)
            logit_params = logit_model.params
        except:
            logit_params = None
        
        models[m] = {
            "gamma": (shape, scale),
            "threshold": threshold,
            "gpd": (c, scale_gpd),
            "p_exceed": np.mean(vcsn_vals > threshold),
            "logit_params": logit_params
        }
        
        print(f"  Month {m}: threshold={threshold:.1f}mm, n_exceed={len(exceedances)}, p_exceed={models[m]['p_exceed']:.3f}")
    
    return models

# ---------- Simulation function ----------
def simulate_monthly_extension(station_name: str, models: dict, 
                               gauges_monthly: pd.DataFrame, 
                               vcsn_monthly: pd.DataFrame):
    """
    Simulate MONTHLY rainfall for the missing period (pre-VCSN only: 1853-1971).
    Uses conditional modeling based on gauge exceedances.
    """
    # Get VCSN start date and create extension period
    vcsn_start = vcsn_monthly["YearMonth"].min()
    gauge_start = gauges_monthly["YearMonth"].min()
    
    # Create dataframe for extension period only (before VCSN starts)
    extension_months = pd.DataFrame({
        "YearMonth": pd.date_range(gauge_start, vcsn_start, freq='MS', inclusive='left')
    })
    extension_months["Year"] = extension_months["YearMonth"].dt.year
    extension_months["Month"] = extension_months["YearMonth"].dt.month
    
    # Merge with gauge data
    extension_months = pd.merge(
        extension_months,
        gauges_monthly[["YearMonth", "Hunua_Rainfall", "Waitakere_Rainfall"]],
        on="YearMonth",
        how="left"
    )
    
    print(f"\nSimulating {len(extension_months)} extension months for {station_name} ({gauge_start.strftime('%Y-%m')} to {vcsn_start.strftime('%Y-%m')})")
    
    simulated_rainfall = []
    
    for _, row in tqdm(extension_months.iterrows(), total=len(extension_months), desc=f"Simulating {station_name}"):
        month = row["Month"]
        
        if month not in models:
            simulated_rainfall.append(0.0)
            continue
        
        model = models[month]
        gamma_shape, gamma_scale = model["gamma"]
        threshold = model["threshold"]
        gpd_shape, gpd_scale = model["gpd"]
        logit_params = model["logit_params"]
        
        # Determine if extreme event occurs
        is_extreme = False
        if logit_params is not None and not pd.isna(row["Hunua_Rainfall"]) and not pd.isna(row["Waitakere_Rainfall"]):
            # Calculate gauge thresholds
            hunua_thresh = np.quantile(gauges_monthly[gauges_monthly["Month"] == month]["Hunua_Rainfall"].dropna(), THRESH_QUANTILE)
            waitakere_thresh = np.quantile(gauges_monthly[gauges_monthly["Month"] == month]["Waitakere_Rainfall"].dropna(), THRESH_QUANTILE)
            
            hunua_exceed = int(row["Hunua_Rainfall"] > hunua_thresh)
            waitakere_exceed = int(row["Waitakere_Rainfall"] > waitakere_thresh)
            
            # Logistic probability
            X = np.array([1, hunua_exceed, waitakere_exceed])
            logit = np.dot(X, logit_params)
            p_extreme = 1 / (1 + np.exp(-logit))
            is_extreme = np.random.rand() < p_extreme
        else:
            # Fallback to unconditional probability
            is_extreme = np.random.rand() < model["p_exceed"]
        
        # Sample rainfall
        if is_extreme and not np.isnan(gpd_shape):
            # Sample from GPD tail
            rainfall = threshold + stats.genpareto.rvs(gpd_shape, scale=gpd_scale)
        else:
            # Sample from gamma bulk
            if not np.isnan(gamma_shape):
                rainfall = stats.gamma.rvs(gamma_shape, scale=gamma_scale)
            else:
                rainfall = 0.0
        
        simulated_rainfall.append(max(0, rainfall))
    
    extension_months["Rainfall"] = simulated_rainfall
    
    # Combine extension with existing VCSN monthly data (stitch together)
    combined_monthly = pd.concat([
        extension_months[["YearMonth", "Year", "Month", "Rainfall"]],
        vcsn_monthly[["YearMonth", "Year", "Month", "Rainfall"]]
    ]).sort_values("YearMonth").reset_index(drop=True)
    
    return combined_monthly

# ---------- Disaggregate to daily ----------
def disaggregate_to_daily(monthly_df: pd.DataFrame, vcsn_daily: pd.DataFrame):
    """
    Disaggregate monthly totals to daily values using VCSN climatological patterns.
    For the extension period (pre-1972), use the typical within-month distribution from VCSN.
    For the VCSN period (1972+), just use the original daily values.
    """
    vcsn_start = vcsn_daily["Date"].min()
    
    # For VCSN period: use original daily values
    vcsn_period = vcsn_daily.copy()
    
    # For extension period: disaggregate using climatology
    vcsn_daily = vcsn_daily.copy()
    vcsn_daily["YearMonth"] = vcsn_daily["Date"].values.astype('datetime64[M]')
    vcsn_daily["Day"] = vcsn_daily["Date"].dt.day
    
    # Calculate monthly totals for VCSN
    vcsn_monthly_totals = vcsn_daily.groupby(["Month", "YearMonth"])["Rainfall"].sum().reset_index()
    vcsn_daily = pd.merge(vcsn_daily, vcsn_monthly_totals, on=["Month", "YearMonth"], suffixes=("", "_monthly"))
    
    # Calculate daily proportion of monthly total
    vcsn_daily["proportion"] = vcsn_daily["Rainfall"] / vcsn_daily["Rainfall_monthly"]
    vcsn_daily["proportion"] = vcsn_daily["proportion"].fillna(0)
    
    # Average proportion by month and day (climatology)
    daily_climatology = vcsn_daily.groupby(["Month", "Day"])["proportion"].mean().reset_index()
    
    # Get extension monthly data (before VCSN starts)
    extension_monthly = monthly_df[monthly_df["YearMonth"] < vcsn_start].copy()
    
    # Create daily dates for extension period
    extension_start = extension_monthly["YearMonth"].min()
    extension_end = vcsn_start - pd.Timedelta(days=1)
    extension_dates = pd.date_range(extension_start, extension_end, freq='D')
    
    extension_daily = pd.DataFrame({"Date": extension_dates})
    extension_daily["Year"] = extension_daily["Date"].dt.year
    extension_daily["Month"] = extension_daily["Date"].dt.month
    extension_daily["Day"] = extension_daily["Date"].dt.day
    extension_daily["YearMonth"] = extension_daily["Date"].values.astype('datetime64[M]')
    
    # Merge with monthly totals
    extension_daily = pd.merge(extension_daily, extension_monthly[["YearMonth", "Rainfall"]], on="YearMonth", how="left")
    extension_daily = extension_daily.rename(columns={"Rainfall": "Monthly_Total"})
    
    # Merge with climatology
    extension_daily = pd.merge(extension_daily, daily_climatology, on=["Month", "Day"], how="left")
    extension_daily["proportion"] = extension_daily["proportion"].fillna(1/30)  # Default uniform if missing
    
    # Normalize proportions within each month to sum to 1
    month_prop_sums = extension_daily.groupby("YearMonth")["proportion"].transform("sum")
    extension_daily["proportion_normalized"] = extension_daily["proportion"] / month_prop_sums
    
    # Calculate daily rainfall
    extension_daily["Rainfall"] = extension_daily["Monthly_Total"] * extension_daily["proportion_normalized"]
    extension_daily = extension_daily[["Date", "Year", "Month", "Rainfall"]]
    
    # Combine extension with original VCSN
    full_daily = pd.concat([
        extension_daily,
        vcsn_period[["Date", "Year", "Month", "Rainfall"]]
    ]).sort_values("Date").reset_index(drop=True)
    
    return full_daily

# ---------- Main processing ----------
if __name__ == "__main__":
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    
    print("="*60)
    print("VCSN Stochastic Extension Script")
    print("="*60)
    
    # --- Load gauge DAILY data ---
    print("\n1. Reading gauge DAILY data...")
    hunua_daily = read_and_prepare_daily(GAUGES["Hunua"])
    waitakere_daily = read_and_prepare_daily(GAUGES["Waitakere"])
    
    print(f"  Hunua: {hunua_daily['Date'].min()} to {hunua_daily['Date'].max()}")
    print(f"  Waitakere: {waitakere_daily['Date'].min()} to {waitakere_daily['Date'].max()}")
    
    # Assert same date range
    assert hunua_daily['Date'].min() == waitakere_daily['Date'].min(), "Gauge start dates don't match"
    assert hunua_daily['Date'].max() == waitakere_daily['Date'].max(), "Gauge end dates don't match"
    
    gauge_date_range = pd.date_range(hunua_daily['Date'].min(), hunua_daily['Date'].max(), freq='D')
    print(f"  Gauge date range: {len(gauge_date_range)} days")
    
    # --- Aggregate gauges to MONTHLY ---
    print("\n2. Aggregating gauges to MONTHLY totals...")
    hunua_monthly = to_monthly_totals(hunua_daily).rename(columns={"Rainfall": "Hunua_Rainfall"})
    waitakere_monthly = to_monthly_totals(waitakere_daily).rename(columns={"Rainfall": "Waitakere_Rainfall"})
    
    gauges_monthly = pd.merge(
        hunua_monthly,
        waitakere_monthly,
        on=["YearMonth", "Year", "Month"],
        how="outer"
    ).sort_values("YearMonth")
    
    print(f"  Gauge monthly records: {len(gauges_monthly)} months")
    
    # --- Load and process each VCSN station ---
    for station_name, vcsn_path in VIRTUAL_STATIONS:
        print(f"\n{'='*60}")
        print(f"Processing station: {station_name}")
        print(f"{'='*60}")
        
        # Load VCSN DAILY data
        print("\n3. Reading VCSN DAILY data...")
        vcsn_daily = read_and_prepare_daily(vcsn_path)
        print(f"  VCSN range: {vcsn_daily['Date'].min()} to {vcsn_daily['Date'].max()}")
        print(f"  VCSN days: {len(vcsn_daily)}")
        
        # Aggregate to MONTHLY
        print("\n4. Aggregating VCSN to MONTHLY totals...")
        vcsn_monthly = to_monthly_totals(vcsn_daily)
        print(f"  VCSN monthly records: {len(vcsn_monthly)} months")
        
        # Fit models on MONTHLY data
        print("\n5. Fitting monthly models...")
        models = fit_monthly_models(vcsn_monthly, gauges_monthly, station_name)
        print(f"  Fitted models for {len(models)} months")
        
        # Simulate missing MONTHLY values (extension period only)
        print("\n6. Simulating monthly extension (1853-1971)...")
        extended_monthly = simulate_monthly_extension(
            station_name, models, gauges_monthly, vcsn_monthly
        )
        print(f"  Complete monthly series: {len(extended_monthly)} months")
        print(f"  Range: {extended_monthly['YearMonth'].min()} to {extended_monthly['YearMonth'].max()}")
        
        # Disaggregate to DAILY (extension period only, then stitch with original VCSN)
        print("\n7. Disaggregating extension to DAILY and stitching with VCSN...")
        extended_daily = disaggregate_to_daily(extended_monthly, vcsn_daily)
        print(f"  Complete daily series: {len(extended_daily)} days")
        
        # Verify final date range
        print(f"  Final range: {extended_daily['Date'].min()} to {extended_daily['Date'].max()}")
        print(f"  Gauge range: {gauge_date_range.min()} to {gauge_date_range.max()}")
        
        # Save
        out_csv = OUTPUT_DIR / f"{station_name}_VCSN_extended_1853-2025.csv"
        extended_daily.to_csv(out_csv, index=False)
        print(f"\n8. Saved: {out_csv}")
        
        # Quick stats
        print(f"\n  Summary statistics:")
        print(f"    Total rainfall: {extended_daily['Rainfall'].sum():.1f} mm")
        print(f"    Mean daily: {extended_daily['Rainfall'].mean():.2f} mm")
        print(f"    Max daily: {extended_daily['Rainfall'].max():.1f} mm")
    
    print(f"\n{'='*60}")
    print("✓ All stations processed successfully!")
    print(f"{'='*60}")