"""
AURA — Model Trainer
Run this ONCE before launching dashboard.py:
    python train_model.py

Put this file in the SAME folder as dashboard.py and city_day.csv
"""

import pandas as pd, numpy as np, joblib, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.resolve()
CSV  = BASE / "city_day.csv"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

if not CSV.exists():
    print(f"[ERROR] city_day.csv not found at {CSV}")
    exit(1)

POLLUTANTS = ['PM2.5','PM10','NO','NO2','NOx','CO','SO2','O3','NH3']
FEATURES   = ['PM2.5','PM10','NO','NO2','NOx','CO','SO2','O3','NH3',
               'month','dayofweek','year','is_weekend','quarter',
               'pm25_lag1','pm25_lag2','pm25_lag3','pm25_lag7',
               'aqi_lag1','pm25_roll7','pm25_roll14','no2_roll7']

print("Loading city_day.csv...")
df  = pd.read_csv(CSV)
ncr = df[df['City'].isin(['Delhi','Gurugram'])].copy()
ncr['Date'] = pd.to_datetime(ncr['Date'])
ncr = ncr.sort_values('Date').reset_index(drop=True)
print(f"  Delhi+Gurugram rows: {len(ncr)}")

for col in POLLUTANTS:
    if col in ncr.columns:
        ncr[col] = ncr[col].interpolate(method='linear').ffill().bfill()

ncr['month']     = ncr['Date'].dt.month
ncr['dayofweek'] = ncr['Date'].dt.dayofweek
ncr['year']      = ncr['Date'].dt.year
ncr['is_weekend']= (ncr['dayofweek'] >= 5).astype(int)
ncr['quarter']   = ncr['Date'].dt.quarter
for lag in [1, 2, 3, 7]:
    ncr[f'pm25_lag{lag}'] = ncr['PM2.5'].shift(lag)
    ncr[f'aqi_lag{lag}']  = ncr['AQI'].shift(lag) if 'AQI' in ncr.columns else 0
ncr['pm25_roll7']  = ncr['PM2.5'].rolling(7).mean()
ncr['pm25_roll14'] = ncr['PM2.5'].rolling(14).mean()
ncr['no2_roll7']   = ncr['NO2'].rolling(7).mean()
ncr['pm25_next']   = ncr['PM2.5'].shift(-1)
ncr.dropna(inplace=True)
print(f"  After feature engineering: {len(ncr)} rows, {len(FEATURES)} features")

X = ncr[FEATURES]; y = ncr['pm25_next']
split = int(len(ncr) * 0.80)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

print("Normalizing with MinMaxScaler...")
sc = MinMaxScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc  = sc.transform(X_te)

print("Training Random Forest (300 trees)...")
rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=2,
     max_features='sqrt', n_jobs=-1, random_state=42, oob_score=True)
rf.fit(X_tr_sc, y_tr)

y_pred = rf.predict(X_te_sc)
mae  = mean_absolute_error(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred) ** 0.5
r2   = r2_score(y_te, y_pred)

print(f"\n  MAE  : {mae:.2f} µg/m³")
print(f"  RMSE : {rmse:.2f} µg/m³")
print(f"  R²   : {r2:.4f}")
print(f"  OOB R²: {rf.oob_score_:.4f}")

joblib.dump(rf, OUT / "aura_rf_model.pkl")
joblib.dump(sc, OUT / "aura_scaler.pkl")
with open(OUT / "model_meta.json", "w") as f:
    json.dump({"mae": round(mae,2), "rmse": round(rmse,2),
               "r2": round(r2,4), "oob": round(rf.oob_score_,4),
               "features": FEATURES, "n_rows": len(ncr)}, f, indent=2)

print(f"\n✅ All files saved to: {OUT}")
print("   - aura_rf_model.pkl")
print("   - aura_scaler.pkl")
print("   - model_meta.json")
print("\nNow run:  streamlit run dashboard.py")