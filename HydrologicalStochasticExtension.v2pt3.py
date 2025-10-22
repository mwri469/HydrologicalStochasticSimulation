"""
stochastic_extend_19month.py

Create stochastic extensions for VCSN virtual stations using 19-month rolling sums.
This approach:
 - Fits Weibull (bulk) + GPD (tail) to 19-month rolling rainfall totals
 - Uses temporal structure from Hunua/Waitakere gauges to determine timing of extremes
 - Samples from site-specific distributions (not gauge distributions)
 - Disaggregates 19-month totals to monthly, then to daily
 - Processes all 3 Waikato sites with shared temporal structure
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

DATA_DIR = Path(r"P:\1011941\1011941.3000\WorkingMaterial\01 IPCC6 MODELLING\01 COLLATE DOWNSCALED NIWA DATA\02 OUTPUT\ConvertedVCSNdata")
GAUGE_DIR = Path(r"P:\1011941\1011941.3000\WorkingMaterial\01 IPCC6 MODELLING\01 COLLATE DOWNSCALED NIWA DATA\00 DATA")
OUTPUT_DIR = Path("../../../02 OUTPUT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "upper_waikato_vcsn": DATA_DIR / "UpperWaikato_pr_29940.csv",
    "lower_waikato_vcsn": DATA_DIR / "LowerWaikato_pr_28200.csv",
    "waipa_vcsn": DATA_DIR / "WaipaRiver_pr_29278.csv",
    "hunua_record": GAUGE_DIR / "PD00_HunuaRanges_RF_1853-2025.csv",
    "waitakere_record": GAUGE_DIR / "PD00_WaitakereRanges_RF_1853-2025.csv"
}

VIRTUAL_STATIONS = [
    ("UpperWaikato", FILES["upper_waikato_vcsn"]),
    ("LowerWaikato", FILES["lower_waikato_vcsn"]),
    ("WaipaRiver", FILES["waipa_vcsn"])
]

ROLLING_WINDOW = 19
THRESH_QUANTILE = 0.85
MIN_EXCEEDANCES = 10
MIN_BULK_SAMPLES = 30

def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent date and rainfall columns."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    rain_col = df.columns[-1]
    df["Rainfall"] = df[rain_col].astype(float)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df[["Date", "Year", "Month", "Rainfall"]]

def read_and_prepare_daily(path: Path) -> pd.DataFrame:
    """Read and process daily rainfall CSV."""
    df = pd.read_csv(path)
    return process_datetime(df)

def to_monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to monthly totals."""
    monthly = df.groupby(["Year", "Month"], as_index=False)["Rainfall"].sum()
    monthly["YearMonth"] = pd.to_datetime(monthly[["Year", "Month"]].assign(day=1))
    return monthly[["YearMonth", "Year", "Month", "Rainfall"]]

def compute_rolling_19month(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 19-month rolling sums from monthly data."""
    monthly_df = monthly_df.sort_values("YearMonth").reset_index(drop=True)
    monthly_df["Rolling_19m"] = monthly_df["Rainfall"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum()
    return monthly_df.dropna(subset=["Rolling_19m"]).reset_index(drop=True)

def fit_rolling_models(rolling_df: pd.DataFrame, station_name: str):
    """
    Fit empirical bulk + GPD tail to 19-month rolling sums.
    Returns fitted parameters and threshold.
    """
    values = rolling_df["Rolling_19m"].dropna().values
    
    if len(values) < MIN_BULK_SAMPLES:
        raise ValueError(f"Insufficient data for {station_name}: {len(values)} samples")
    
    threshold = np.quantile(values, THRESH_QUANTILE)
    
    bulk = values[values <= threshold]
    if len(bulk) < MIN_BULK_SAMPLES:
        raise ValueError(f"Insufficient bulk samples for {station_name}")
    
    exceedances = values[values > threshold] - threshold
    if len(exceedances) >= MIN_EXCEEDANCES:
        gpd_shape, gpd_loc, gpd_scale = stats.genpareto.fit(exceedances, floc=0)
    else:
        print(f"  Warning: Only {len(exceedances)} exceedances, GPD may be unreliable")
        gpd_shape, gpd_scale = 0.1, np.std(exceedances) if len(exceedances) > 0 else 100
    
    model = {
        "bulk_empirical": bulk.copy(),
        "threshold": threshold,
        "gpd": (gpd_shape, gpd_scale),
        "p_exceed": np.mean(values > threshold),
        "n_samples": len(values),
        "n_exceedances": len(exceedances)
    }
    
    print(f"  {station_name}: threshold={threshold:.1f}mm, "
          f"p_exceed={model['p_exceed']:.3f}, n_exceed={len(exceedances)}, "
          f"n_bulk={len(bulk)}")
    
    return model

def compute_gauge_extremeness(gauge_rolling: pd.DataFrame) -> pd.Series:
    """
    Compute 'extremeness' score for gauge based on its 19-month rolling values.
    Returns normalized score (0-1) where 1 = most extreme.
    """
    values = gauge_rolling["Rolling_19m"].values
    ranks = stats.rankdata(values) / len(values)
    gauge_rolling = gauge_rolling.copy()
    gauge_rolling["extremeness"] = ranks
    return gauge_rolling

def simulate_with_temporal_structure(station_models: dict, gauge_rolling: pd.DataFrame, 
                                     extension_months: pd.DataFrame, noise_scale: float = 0.2):
    """
    Simulate 19-month totals for extension period using gauge temporal structure.
    
    Args:
        station_models: Dict of {station_name: fitted_model}
        gauge_rolling: Gauge 19-month rolling with extremeness scores
        extension_months: DataFrame of months to simulate
        noise_scale: Amount of noise to add to gauge extremeness signal (0-1)
    
    Returns:
        Dict of {station_name: simulated_19month_series}
    """
    extension_start = extension_months["YearMonth"].min()
    extension_end = extension_months["YearMonth"].max()
    
    gauge_ext = gauge_rolling[
        (gauge_rolling["YearMonth"] >= extension_start) & 
        (gauge_rolling["YearMonth"] <= extension_end)
    ].copy()
    
    if len(gauge_ext) == 0:
        raise ValueError("No gauge data overlaps with extension period")
    
    simulated = {}
    
    for station_name, model in station_models.items():
        bulk_empirical = model["bulk_empirical"]
        threshold = model["threshold"]
        gpd_shape, gpd_scale = model["gpd"]
        p_exceed = model["p_exceed"]
        
        station_values = []
        
        for _, row in gauge_ext.iterrows():
            gauge_extreme = row["extremeness"]
            
            noisy_extreme = np.clip(gauge_extreme + np.random.normal(0, noise_scale), 0, 1)
            
            adjusted_p_exceed = p_exceed * (0.5 + noisy_extreme)
            adjusted_p_exceed = np.clip(adjusted_p_exceed, 0, 0.5)
            
            is_extreme = np.random.rand() < adjusted_p_exceed
            
            if is_extreme and not np.isnan(gpd_shape):
                value = threshold + stats.genpareto.rvs(gpd_shape, scale=gpd_scale)
            else:
                value = np.random.choice(bulk_empirical)
            
            station_values.append(max(0, value))
        
        simulated[station_name] = pd.DataFrame({
            "YearMonth": gauge_ext["YearMonth"].values,
            "Rolling_19m_simulated": station_values
        })
    
    return simulated

def disaggregate_19month_to_monthly(rolling_19m_series: pd.DataFrame, 
                                   vcsn_monthly: pd.DataFrame,
                                   preserve_seasonality: bool = True) -> pd.DataFrame:
    """
    Disaggregate 19-month totals to monthly values by resampling actual 
    19-month patterns from VCSN and scaling to match simulated totals.
    """
    vcsn_monthly = vcsn_monthly.sort_values("YearMonth").reset_index(drop=True)
    vcsn_monthly["Rolling_19m"] = vcsn_monthly["Rainfall"].rolling(
        window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW
    ).sum()
    
    vcsn_with_rolling = vcsn_monthly.dropna(subset=["Rolling_19m"]).reset_index(drop=True)
    
    if len(vcsn_with_rolling) < ROLLING_WINDOW:
        raise ValueError("Insufficient VCSN data for pattern resampling")
    
    extension_start = rolling_19m_series["YearMonth"].min() - pd.DateOffset(months=ROLLING_WINDOW-1)
    extension_end = rolling_19m_series["YearMonth"].max() + pd.DateOffset(months=ROLLING_WINDOW-1)
    
    monthly_dates = pd.date_range(extension_start, extension_end, freq='MS')
    monthly_extended = pd.DataFrame({
        "YearMonth": monthly_dates,
        "Month": monthly_dates.month,
        "Year": monthly_dates.year
    })
    
    monthly_extended["Rainfall"] = np.nan
    
    for idx, row in rolling_19m_series.iterrows():
        target_ym = row["YearMonth"]
        target_total = row["Rolling_19m_simulated"]
        
        candidate_indices = vcsn_with_rolling.index[
            (vcsn_with_rolling["Rolling_19m"] > target_total * 0.7) &
            (vcsn_with_rolling["Rolling_19m"] < target_total * 1.3)
        ].tolist()
        
        if len(candidate_indices) < 5:
            candidate_indices = vcsn_with_rolling.index[
                (vcsn_with_rolling["Rolling_19m"] > target_total * 0.5) &
                (vcsn_with_rolling["Rolling_19m"] < target_total * 1.5)
            ].tolist()
        
        if len(candidate_indices) == 0:
            candidate_indices = vcsn_with_rolling.index.tolist()
        
        if preserve_seasonality:
            target_month = target_ym.month
            month_candidates = [
                i for i in candidate_indices 
                if vcsn_with_rolling.loc[i, "Month"] in [target_month, (target_month % 12) + 1, ((target_month - 2) % 12) + 1]
            ]
            if len(month_candidates) >= 3:
                candidate_indices = month_candidates
        
        selected_idx = np.random.choice(candidate_indices)
        
        vcsn_sequence_start = vcsn_with_rolling.loc[selected_idx, "YearMonth"] - pd.DateOffset(months=ROLLING_WINDOW-1)
        vcsn_sequence_indices = vcsn_monthly[
            (vcsn_monthly["YearMonth"] >= vcsn_sequence_start) &
            (vcsn_monthly["YearMonth"] < vcsn_sequence_start + pd.DateOffset(months=ROLLING_WINDOW))
        ].index.tolist()
        
        if len(vcsn_sequence_indices) != ROLLING_WINDOW:
            continue
        
        vcsn_pattern = vcsn_monthly.loc[vcsn_sequence_indices, "Rainfall"].values
        observed_total = vcsn_pattern.sum()
        
        if observed_total > 0:
            scaled_pattern = vcsn_pattern * (target_total / observed_total)
        else:
            scaled_pattern = np.full(ROLLING_WINDOW, target_total / ROLLING_WINDOW)
        
        target_start = target_ym - pd.DateOffset(months=ROLLING_WINDOW-1)
        target_indices = monthly_extended[
            (monthly_extended["YearMonth"] >= target_start) &
            (monthly_extended["YearMonth"] <= target_ym)
        ].index.tolist()
        
        if len(target_indices) == ROLLING_WINDOW:
            for i, target_idx in enumerate(target_indices):
                if pd.isna(monthly_extended.loc[target_idx, "Rainfall"]):
                    monthly_extended.loc[target_idx, "Rainfall"] = scaled_pattern[i]
    
    monthly_extended["Rainfall"] = monthly_extended["Rainfall"].fillna(
        monthly_extended.groupby("Month")["Rainfall"].transform("mean")
    )
    
    return monthly_extended[["YearMonth", "Year", "Month", "Rainfall"]]

def disaggregate_to_daily(monthly_df: pd.DataFrame, vcsn_daily: pd.DataFrame):
    """Disaggregate monthly totals to daily using VCSN climatology."""
    vcsn_start = vcsn_daily["Date"].min()
    
    vcsn_period = vcsn_daily.copy()
    
    vcsn_daily = vcsn_daily.copy()
    vcsn_daily["YearMonth"] = vcsn_daily["Date"].values.astype('datetime64[M]')
    vcsn_daily["Day"] = vcsn_daily["Date"].dt.day
    
    vcsn_monthly_totals = vcsn_daily.groupby(["Month", "YearMonth"])["Rainfall"].sum().reset_index()
    vcsn_daily = pd.merge(vcsn_daily, vcsn_monthly_totals, on=["Month", "YearMonth"], suffixes=("", "_monthly"))
    
    vcsn_daily["proportion"] = vcsn_daily["Rainfall"] / vcsn_daily["Rainfall_monthly"]
    vcsn_daily["proportion"] = vcsn_daily["proportion"].fillna(0)
    
    daily_climatology = vcsn_daily.groupby(["Month", "Day"])["proportion"].mean().reset_index()
    
    extension_monthly = monthly_df[monthly_df["YearMonth"] < vcsn_start].copy()
    
    if len(extension_monthly) == 0:
        return vcsn_period[["Date", "Year", "Month", "Rainfall"]]
    
    extension_start = extension_monthly["YearMonth"].min()
    extension_end = vcsn_start - pd.Timedelta(days=1)
    extension_dates = pd.date_range(extension_start, extension_end, freq='D')
    
    extension_daily = pd.DataFrame({"Date": extension_dates})
    extension_daily["Year"] = extension_daily["Date"].dt.year
    extension_daily["Month"] = extension_daily["Date"].dt.month
    extension_daily["Day"] = extension_daily["Date"].dt.day
    extension_daily["YearMonth"] = extension_daily["Date"].values.astype('datetime64[M]')
    
    extension_daily = pd.merge(extension_daily, extension_monthly[["YearMonth", "Rainfall"]], on="YearMonth", how="left")
    extension_daily = extension_daily.rename(columns={"Rainfall": "Monthly_Total"})
    
    extension_daily = pd.merge(extension_daily, daily_climatology, on=["Month", "Day"], how="left")
    extension_daily["proportion"] = extension_daily["proportion"].fillna(1/30)
    
    month_prop_sums = extension_daily.groupby("YearMonth")["proportion"].transform("sum")
    extension_daily["proportion_normalized"] = extension_daily["proportion"] / month_prop_sums
    
    extension_daily["Rainfall"] = extension_daily["Monthly_Total"] * extension_daily["proportion_normalized"]
    extension_daily = extension_daily[["Date", "Year", "Month", "Rainfall"]]
    
    full_daily = pd.concat([
        extension_daily,
        vcsn_period[["Date", "Year", "Month", "Rainfall"]]
    ]).sort_values("Date").reset_index(drop=True)
    
    return full_daily

if __name__ == "__main__":
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    
    print("="*60)
    print("VCSN 19-Month Rolling Stochastic Extension")
    print("="*60)
    
    print("\n1. Reading gauge daily data...")
    hunua_daily = read_and_prepare_daily(FILES["hunua_record"])
    waitakere_daily = read_and_prepare_daily(FILES["waitakere_record"])
    
    print(f"  Hunua: {hunua_daily['Date'].min()} to {hunua_daily['Date'].max()}")
    print(f"  Waitakere: {waitakere_daily['Date'].min()} to {waitakere_daily['Date'].max()}")
    
    print("\n2. Aggregating to monthly and computing 19-month rolling sums...")
    hunua_monthly = to_monthly_totals(hunua_daily)
    waitakere_monthly = to_monthly_totals(waitakere_daily)
    
    hunua_rolling = compute_rolling_19month(hunua_monthly)
    waitakere_rolling = compute_rolling_19month(waitakere_monthly)
    
    gauge_rolling = pd.merge(
        hunua_rolling[["YearMonth", "Rolling_19m"]],
        waitakere_rolling[["YearMonth", "Rolling_19m"]],
        on="YearMonth",
        suffixes=("_hunua", "_waitakere")
    )
    gauge_rolling["Rolling_19m"] = (gauge_rolling["Rolling_19m_hunua"] + 
                                     gauge_rolling["Rolling_19m_waitakere"]) / 2
    
    gauge_rolling = compute_gauge_extremeness(gauge_rolling)
    
    print(f"  Gauge 19-month rolling: {len(gauge_rolling)} periods")
    
    print("\n3. Loading VCSN data and fitting models...")
    station_models = {}
    vcsn_data = {}
    
    for station_name, vcsn_path in VIRTUAL_STATIONS:
        print(f"\n  Processing {station_name}...")
        vcsn_daily = read_and_prepare_daily(vcsn_path)
        vcsn_monthly = to_monthly_totals(vcsn_daily)
        vcsn_rolling = compute_rolling_19month(vcsn_monthly)
        
        print(f"    VCSN range: {vcsn_daily['Date'].min()} to {vcsn_daily['Date'].max()}")
        print(f"    19-month periods: {len(vcsn_rolling)}")
        
        model = fit_rolling_models(vcsn_rolling, station_name)
        station_models[station_name] = model
        vcsn_data[station_name] = {
            "daily": vcsn_daily,
            "monthly": vcsn_monthly,
            "rolling": vcsn_rolling
        }
    
    print("\n4. Simulating extension period with temporal structure...")
    vcsn_start = min(data["monthly"]["YearMonth"].min() for data in vcsn_data.values())
    gauge_start = gauge_rolling["YearMonth"].min()
    
    extension_months = pd.DataFrame({
        "YearMonth": pd.date_range(gauge_start, vcsn_start, freq='MS', inclusive='left')
    })
    extension_months["Year"] = extension_months["YearMonth"].dt.year
    extension_months["Month"] = extension_months["YearMonth"].dt.month
    
    print(f"  Extension period: {extension_months['YearMonth'].min()} to {extension_months['YearMonth'].max()}")
    print(f"  Simulating {len(extension_months)} months across {len(VIRTUAL_STATIONS)} stations...")
    
    simulated_19m = simulate_with_temporal_structure(
        station_models, gauge_rolling, extension_months, noise_scale=0.2
    )
    
    print("\n5. Disaggregating and saving results...")
    for station_name in simulated_19m.keys():
        print(f"\n  {station_name}:")
        
        monthly_extended = disaggregate_19month_to_monthly(
            simulated_19m[station_name],
            vcsn_data[station_name]["monthly"]
        )
        
        combined_monthly = pd.concat([
            monthly_extended,
            vcsn_data[station_name]["monthly"]
        ]).sort_values("YearMonth").drop_duplicates(subset=["YearMonth"]).reset_index(drop=True)
        
        print(f"    Combined monthly: {len(combined_monthly)} months")
        
        extended_daily = disaggregate_to_daily(
            combined_monthly,
            vcsn_data[station_name]["daily"]
        )
        
        print(f"    Final daily range: {extended_daily['Date'].min()} to {extended_daily['Date'].max()}")
        print(f"    Total days: {len(extended_daily)}")
        
        out_csv = OUTPUT_DIR / f"{station_name}_VCSN_19m_extended_1853-2025.csv"
        extended_daily.to_csv(out_csv, index=False)
        print(f"    Saved: {out_csv}")
        
        print(f"    Summary: Total={extended_daily['Rainfall'].sum():.1f}mm, "
              f"Mean={extended_daily['Rainfall'].mean():.2f}mm, "
              f"Max={extended_daily['Rainfall'].max():.1f}mm")
    
    print(f"\n{'='*60}")
    print("✓ All stations processed!")
    print(f"{'='*60}")
