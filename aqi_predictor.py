"""
╔══════════════════════════════════════════════════════════╗
║  AURA — Module 2: Random Forest AQI Predictor           ║
║  Reads TDI from Module 1 CSV + CPCB AQI patterns        ║
╚══════════════════════════════════════════════════════════╝

Real Dataset Sources:
  CPCB Hourly AQI → https://cpcb.nic.in/automatic-monitoring-data/
  data.gov.in     → https://data.gov.in/resource/real-time-air-quality-index
  AQICN token     → https://aqicn.org/data-platform/token/

If you have a real CPCB CSV, set: CPCB_CSV = "path/to/your_cpcb_data.csv"
Expected columns: datetime, pm25, station, city
"""

import pandas as pd
import numpy as np
import joblib
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ── Paths ─────────────────────────────────────────────────────────────────
TDI_CSV    = Path("../outputs/tdi_results.csv")   # from Module 1
CPCB_CSV   = Path("../data/city_day.csv")
MODEL_OUT  = Path("../outputs/aura_rf_model.pkl")
SCALER_OUT = Path("../outputs/aura_scaler.pkl")
META_OUT   = Path("../outputs/model_meta.json")
PLOT_DIR   = Path("../outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "tdi", "tdi_lag1", "tdi_lag2",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag6",
    "pm25_roll4h", "pm25_roll8h",
    "hour", "hour_sin", "hour_cos",          # cyclic encoding
    "dayofweek", "is_weekend", "month",
    "wind_kmh", "humidity", "temp_c",
]
TARGET = "pm25_next"


# ══════════════════════════════════════════════════════════════════════════
# 1.  LOAD REAL CPCB DATA (if available) or generate synthetic
# ══════════════════════════════════════════════════════════════════════════
def load_or_generate_aqi(n_days=365) -> pd.DataFrame:
    """Load real CPCB CSV or generate synthetic Delhi AQI data."""

    if CPCB_CSV and Path(CPCB_CSV).exists():
        print(f"📂 Loading real CPCB data from {CPCB_CSV} …")
        df = pd.read_csv(CPCB_CSV, parse_dates=["datetime"])
        df = df.rename(columns={"datetime":"timestamp", "pm25":"pm25_raw"})
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["pm25"] = df["pm25_raw"].clip(10, 500).fillna(method="ffill")
        print(f"   Loaded {len(df):,} rows  "
              f"| PM2.5: {df['pm25'].min():.0f}–{df['pm25'].max():.0f} µg/m³")
        return df

    # ── Synthetic generation (matches CPCB Delhi 2022-23 patterns) ──
    print("📊 Generating synthetic CPCB-style AQI dataset (Delhi 2022-23) …")
    np.random.seed(42)
    timestamps = pd.date_range("2022-10-01", periods=n_days*24, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"]     = df["timestamp"].dt.month
    df["is_weekend"]= (df["dayofweek"] >= 5).astype(int)

    # Seasonal baseline (Delhi stubble burning peaks Oct-Dec)
    seasonal = {10:165, 11:210, 12:230, 1:215, 2:175, 3:140,
                4:100,  5: 85,  6: 75,  7: 72, 8: 78, 9:105}
    df["seasonal"] = df["month"].map(seasonal)

    # Diurnal pattern (AQI rises with morning & evening traffic)
    diurnal = np.array([
        0.75,0.70,0.65,0.62,0.65,0.80,
        1.05,1.35,1.45,1.30,1.15,1.05,
        1.00,0.98,1.00,1.08,1.20,1.40,
        1.45,1.35,1.20,1.05,0.90,0.80
    ])
    df["diurnal"] = df["hour"].map(dict(enumerate(diurnal)))

    # Weather (Delhi monsoon: Jun-Sep low wind; winter: Dec-Feb high pollution)
    month_wind = {10:6,11:4,12:3,1:3,2:4,3:7,4:9,5:11,6:9,7:8,8:7,9:7}
    df["wind_kmh"]  = df["month"].map(month_wind) + np.random.exponential(3, len(df))
    df["wind_kmh"]  = df["wind_kmh"].clip(0, 35).round(1)
    df["humidity"]  = (65 + np.random.normal(0,12,len(df))).clip(20,100).round(1)
    df["temp_c"]    = (22 - df["month"].map({12:10,1:10,2:8,3:2,4:-2,5:-8,6:-5,7:-3,8:-2,9:-1,10:2,11:6}).fillna(0)
                       + np.random.normal(0,3,len(df))).round(1)

    # PM2.5 = seasonal × diurnal × traffic − wind dispersion + noise
    df["pm25"] = (
        df["seasonal"] * df["diurnal"]
        - df["wind_kmh"] * 2.5
        + np.random.normal(0, 20, len(df))
    ).clip(15, 480).round(1)

    print(f"   Generated {len(df):,} rows  "
          f"| PM2.5: {df['pm25'].min():.0f}–{df['pm25'].max():.0f} µg/m³")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2.  LOAD TDI FROM MODULE 1 AND MERGE
# ══════════════════════════════════════════════════════════════════════════
def load_tdi() -> pd.DataFrame:
    """Load TDI CSV output from Module 1."""
    if TDI_CSV.exists():
        print(f"📂 Loading TDI data from {TDI_CSV} …")
        tdi_df = pd.read_csv(TDI_CSV)
        if "timestamp" in tdi_df.columns:
            tdi_df["timestamp"] = pd.to_datetime(tdi_df["timestamp"])
        return tdi_df
    else:
        # Generate fallback TDI from frame_summary
        summary = Path("../data/frame_summary.csv")
        if summary.exists():
            df = pd.read_csv(summary)
            print(f"📂 Loaded {len(df)} TDI records from frame_summary.csv")
            return df
        print("[WARN] No TDI CSV found — synthesizing TDI column")
        return None


# ══════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, tdi_df=None) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Merge TDI if available
    if tdi_df is not None and "tdi" in tdi_df.columns:
        # Use per-hour average TDI from detector output
        if "hour" in tdi_df.columns:
            hourly_tdi = tdi_df.groupby("hour")["tdi"].mean().to_dict()
            df["tdi"] = df["hour"].map(hourly_tdi).fillna(25.0)
        else:
            df["tdi"] = 25.0
    else:
        # Synthesize TDI from hour pattern
        hour_tdi = {
            0:5, 1:3, 2:3, 3:3, 4:5, 5:10,
            6:18, 7:32, 8:42, 9:38, 10:28, 11:22,
            12:24, 13:22, 14:20, 15:25, 16:35, 17:48,
            18:52, 19:44, 20:30, 21:20, 22:13, 23:8
        }
        df["tdi"] = df["hour"].map(hour_tdi) + np.random.normal(0, 5, len(df))
        df["tdi"] = df["tdi"].clip(1, 120)

    # Lag features
    df["tdi_lag1"]    = df["tdi"].shift(1)
    df["tdi_lag2"]    = df["tdi"].shift(2)
    df["pm25_lag1"]   = df["pm25"].shift(1)
    df["pm25_lag2"]   = df["pm25"].shift(2)
    df["pm25_lag3"]   = df["pm25"].shift(3)
    df["pm25_lag6"]   = df["pm25"].shift(6)

    # Rolling statistics
    df["pm25_roll4h"] = df["pm25"].rolling(4).mean()
    df["pm25_roll8h"] = df["pm25"].rolling(8).mean()

    # Cyclic hour encoding (prevents 23→0 discontinuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Target: PM2.5 one hour ahead
    df["pm25_next"] = df["pm25"].shift(-1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"   Feature engineering done → {len(df):,} rows, {len(FEATURES)} features")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 4.  TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════
def train_model(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df[TARGET].values

    # Time-series aware split (no shuffle)
    split = int(len(df) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    print(f"\n🌲 Training Random Forest  (train={len(X_tr):,}  test={len(X_te):,}) …")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )
    rf.fit(X_tr_sc, y_tr)

    y_pred = rf.predict(X_te_sc)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    oob  = rf.oob_score_

    print(f"\n{'═'*50}")
    print(f"  Random Forest — Test Set Results")
    print(f"  MAE   : {mae:.2f} µg/m³")
    print(f"  RMSE  : {rmse:.2f} µg/m³")
    print(f"  R²    : {r2:.4f}")
    print(f"  OOB R²: {oob:.4f}")
    print(f"{'═'*50}")

    # Feature importances
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\n  Top-8 Feature Importances:")
    for feat, imp in fi.head(8).items():
        bar = "█" * int(imp * 80)
        print(f"    {feat:<18} {bar}  {imp:.4f}")

    # Save
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)

    meta = {"mae": round(mae,2), "rmse": round(rmse,2), "r2": round(r2,4),
            "oob_r2": round(oob,4), "n_train": len(X_tr),
            "n_test": len(X_te), "features": FEATURES,
            "trained_at": datetime.now().isoformat()}
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Model  → {MODEL_OUT}")
    print(f"  ✅ Scaler → {SCALER_OUT}")
    print(f"  ✅ Meta   → {META_OUT}")

    return rf, scaler, y_te, y_pred, fi, df[split:]


# ══════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════
def plot_results(y_te, y_pred, fi, test_df):
    fig = plt.figure(figsize=(18, 14), facecolor="#04060f")
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    def ax_style(ax, title):
        ax.set_facecolor("#0b1022")
        ax.set_title(title, color="#00e5ff", fontsize=11, pad=8)
        ax.spines[:].set_color("#1e2a40")
        ax.tick_params(colors="#6b7a99")
        ax.xaxis.label.set_color("#e8eaf6")
        ax.yaxis.label.set_color("#e8eaf6")
        ax.grid(alpha=0.1, color="#1e2a40")

    # 1. Actual vs Predicted (time series)
    ax1 = fig.add_subplot(gs[0, :])
    N = min(500, len(y_te))
    x_ax = range(N)
    ax1.plot(x_ax, list(y_te)[:N],   color="#ff4444", lw=1.5, label="Actual PM2.5",    alpha=0.9)
    ax1.plot(x_ax, list(y_pred)[:N], color="#00e5ff", lw=1.5, label="Predicted PM2.5", alpha=0.85, ls="--")
    ax1.fill_between(x_ax,
                     [v-10 for v in list(y_pred)[:N]],
                     [v+10 for v in list(y_pred)[:N]],
                     alpha=0.1, color="#00e5ff")
    ax1.axhline(200, color="#ffaa00", lw=0.8, ls=":", alpha=0.7, label="Poor AQI (200)")
    ax1.axhline(300, color="#ff6b35", lw=0.8, ls=":", alpha=0.7, label="Very Poor (300)")
    ax_style(ax1, "Actual vs Predicted PM2.5 — Test Set (first 500 hours)")
    ax1.set_ylabel("PM2.5 µg/m³")
    ax1.legend(facecolor="#0b1022", edgecolor="#1e2a40", labelcolor="#e8eaf6",
               fontsize=8, loc="upper right")

    # 2. Feature Importance bar
    ax2 = fig.add_subplot(gs[1, 0])
    top_fi = fi.head(12)
    colors = ["#00e5ff" if i < 3 else "#7fff6e" if i < 6 else "#6b7a99"
              for i in range(len(top_fi))]
    bars = ax2.barh(top_fi.index[::-1], top_fi.values[::-1], color=colors[::-1])
    ax2.set_xlabel("Importance Score")
    ax_style(ax2, "Feature Importances (Random Forest)")
    for bar, val in zip(bars, top_fi.values[::-1]):
        ax2.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", ha="left", color="#6b7a99", fontsize=7)

    # 3. Residuals histogram
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = np.array(list(y_te)) - np.array(list(y_pred))
    ax3.hist(residuals, bins=70, color="#7fff6e", edgecolor="none", alpha=0.7)
    ax3.axvline(0, color="#ff4444", lw=1.5)
    ax3.axvline(residuals.mean(), color="#ffaa00", lw=1, ls="--",
                label=f"Mean={residuals.mean():.1f}")
    ax3.set_xlabel("Residual µg/m³")
    ax3.set_ylabel("Frequency")
    ax_style(ax3, "Prediction Error Distribution")
    ax3.legend(facecolor="#0b1022", labelcolor="#e8eaf6", fontsize=8)

    # 4. Scatter: Actual vs Predicted
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(list(y_te), list(y_pred), alpha=0.25, s=8, color="#00e5ff")
    lim = [min(list(y_te)+list(y_pred)), max(list(y_te)+list(y_pred))]
    ax4.plot(lim, lim, color="#ff4444", lw=1.5, label="Perfect prediction")
    ax4.set_xlabel("Actual PM2.5")
    ax4.set_ylabel("Predicted PM2.5")
    ax_style(ax4, "Scatter: Actual vs Predicted")
    mae_val = mean_absolute_error(y_te, y_pred)
    r2_val  = r2_score(y_te, y_pred)
    ax4.text(0.05, 0.92, f"MAE={mae_val:.1f} µg/m³   R²={r2_val:.3f}",
             transform=ax4.transAxes, color="#00e5ff", fontsize=8)
    ax4.legend(facecolor="#0b1022", labelcolor="#e8eaf6", fontsize=8)

    # 5. TDI vs PM2.5 correlation
    ax5 = fig.add_subplot(gs[2, 1])
    if "tdi" in test_df.columns:
        ax5.scatter(test_df["tdi"].values[:N], list(y_te)[:N],
                    alpha=0.3, s=8, color="#ff6b35")
        ax5.set_xlabel("Traffic Density Index (TDI)")
        ax5.set_ylabel("PM2.5 µg/m³")
        ax_style(ax5, "TDI → PM2.5 Correlation (Core AURA Insight)")
        # Add trendline
        z = np.polyfit(test_df["tdi"].values[:N], list(y_te)[:N], 1)
        p = np.poly1d(z)
        xr = np.linspace(test_df["tdi"].min(), test_df["tdi"].max(), 100)
        ax5.plot(xr, p(xr), color="#00e5ff", lw=2, label=f"y={z[0]:.1f}x+{z[1]:.0f}")
        ax5.legend(facecolor="#0b1022", labelcolor="#e8eaf6", fontsize=8)

    out = PLOT_DIR / "aura_model_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#04060f")
    print(f"\n  📊 Full plot → {out}")
    plt.close()
    return out


# ══════════════════════════════════════════════════════════════════════════
# 6.  LIVE INFERENCE FUNCTION (imported by Streamlit dashboard)
# ══════════════════════════════════════════════════════════════════════════
def predict_aqi(tdi: float, pm25_history: list,
                hour: int, wind_kmh=8.0, humidity=65.0, temp_c=22.0) -> dict:
    """
    Live inference — called by Streamlit every tick.
    pm25_history: list of last 8 PM2.5 readings (most recent last)
    Returns predicted PM2.5 + AQI category + recommended action
    """
    if not MODEL_OUT.exists():
        # Fallback: linear estimate
        base = 160 if (7<=hour<=10 or 17<=hour<=20) else 110
        pred = base + tdi * 1.1
        return {"pm25_pred": round(pred,1), "category": "Moderate",
                "action": "Model not trained — run aqi_predictor.py first",
                "reroute": pred > 180}

    model  = joblib.load(MODEL_OUT)
    scaler = joblib.load(SCALER_OUT)

    while len(pm25_history) < 8:
        pm25_history = [pm25_history[0]] + list(pm25_history)

    row = {
        "tdi":         tdi,
        "tdi_lag1":    tdi * 0.91,
        "tdi_lag2":    tdi * 0.83,
        "pm25_lag1":   pm25_history[-1],
        "pm25_lag2":   pm25_history[-2],
        "pm25_lag3":   pm25_history[-3],
        "pm25_lag6":   pm25_history[-6],
        "pm25_roll4h": np.mean(pm25_history[-4:]),
        "pm25_roll8h": np.mean(pm25_history[-8:]),
        "hour":        hour,
        "hour_sin":    np.sin(2*np.pi*hour/24),
        "hour_cos":    np.cos(2*np.pi*hour/24),
        "dayofweek":   datetime.now().weekday(),
        "is_weekend":  int(datetime.now().weekday() >= 5),
        "month":       datetime.now().month,
        "wind_kmh":    wind_kmh,
        "humidity":    humidity,
        "temp_c":      temp_c,
    }

    X = np.array([[row[f] for f in FEATURES]])
    pred = float(model.predict(scaler.transform(X))[0])

    # CPCB scale
    if pred <= 50:    cat, action, reroute = "Good",        "No action",           False
    elif pred <= 100: cat, action, reroute = "Satisfactory","Monitor",             False
    elif pred <= 200: cat, action, reroute = "Moderate",    "Alert sensitive groups",False
    elif pred <= 300: cat, action, reroute = "Poor",        "Consider rerouting",  True
    elif pred <= 400: cat, action, reroute = "Very Poor",   "Reroute immediately", True
    else:             cat, action, reroute = "Severe",      "Emergency protocol",  True

    return {"pm25_pred": round(pred,1), "category": cat,
            "action": action, "reroute": reroute}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  AURA — Module 2: AQI Prediction Model")
    print("=" * 55)

    # 1. Load AQI data
    aqi_df = load_or_generate_aqi(n_days=365)

    # 2. Load TDI from Module 1
    tdi_df = load_tdi()

    # 3. Engineer features
    df = engineer_features(aqi_df, tdi_df)

    # 4. Train
    rf, scaler, y_te, y_pred, fi, test_df = train_model(df)

    # 5. Plot
    plot_path = plot_results(y_te, y_pred, fi, test_df)

    # 6. Demo inference
    print(f"\n{'─'*50}")
    print("  Live Inference Demo (morning rush hour):")
    result = predict_aqi(
        tdi=85,
        pm25_history=[145, 155, 162, 170, 175, 180, 185, 190],
        hour=8, wind_kmh=3.5, humidity=72, temp_c=18
    )
    print(f"  TDI Input     : 85 veh/frame  (heavy congestion)")
    print(f"  PM2.5 Predicted: {result['pm25_pred']} µg/m³")
    print(f"  AQI Category  : {result['category']}")
    print(f"  Recommended   : {result['action']}")
    print(f"  Reroute?      : {'⚡ YES' if result['reroute'] else '✓ No'}")