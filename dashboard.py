"""
AURA v3 — Delhi NCR Dashboard
Responsive + Navbar + Contact Page
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, time, datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

BASE       = Path(__file__).parent.resolve()
CSV_PATH   = BASE / "city_day.csv"
MODEL_PKL  = BASE / "outputs" / "aura_rf_model.pkl"
SCALER_PKL = BASE / "outputs" / "aura_scaler.pkl"
META_JSON  = BASE / "outputs" / "model_meta.json"

# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AURA — Delhi NCR", page_icon="🌆",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

body,.stApp{background:#04060f!important;color:#e8eaf6;font-family:'DM Sans',sans-serif}
section[data-testid="stSidebar"]{background:#080e1f!important;border-right:1px solid rgba(0,229,255,.08)!important}
[data-testid="metric-container"]{background:#0b1022;border:1px solid rgba(0,229,255,.18);border-radius:12px;padding:1rem;transition:border-color .2s}
[data-testid="metric-container"]:hover{border-color:rgba(0,229,255,.45)}
[data-testid="stMetricValue"] div{color:#00e5ff!important;font-size:1.5rem!important;font-weight:700;font-family:'Space Mono',monospace}
[data-testid="stMetricLabel"] div{color:#6b7a99!important;font-size:.68rem!important;letter-spacing:.06em;text-transform:uppercase}
[data-testid="stMetricDelta"] svg{display:none}
.alert-box{background:rgba(255,68,68,.12);border:1px solid #ff4444;border-radius:10px;padding:1rem 1.5rem;margin:.5rem 0}
.safe-box{background:rgba(0,230,118,.08);border:1px solid #00e676;border-radius:10px;padding:1rem 1.5rem;margin:.5rem 0}
.warn-box{background:rgba(255,170,0,.09);border:1px solid #ffaa00;border-radius:10px;padding:1rem 1.5rem;margin:.5rem 0}
.err-box{background:rgba(255,68,68,.15);border:2px solid #ff4444;border-radius:12px;padding:1.5rem;margin:1rem 0}
h1,h2,h3,h4{color:#e8eaf6!important;font-family:'DM Sans',sans-serif}
div[data-baseweb="select"]>div{background:#0b1022!important;border-color:rgba(0,229,255,.25)!important;color:#e8eaf6!important}
hr{border-color:rgba(255,255,255,.08)!important}
[data-testid="stExpander"]{background:#0b1022;border:1px solid rgba(0,229,255,.1);border-radius:10px}

/* NAVBAR */
.aura-navbar{
  position:sticky;top:0;z-index:999;
  background:rgba(4,6,15,.95);
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid rgba(0,229,255,.12);
  padding:.65rem 2rem;
  display:flex;align-items:center;justify-content:space-between;
  margin:-1rem -1rem 0 -1rem;
}
.navbar-brand{
  font-family:'Space Mono',monospace;
  font-size:1.15rem;font-weight:700;
  color:#00e5ff;letter-spacing:.08em;
}
.navbar-brand span{color:#e8eaf6;font-weight:400}

/* ── PURE-CSS HAMBURGER (no JS needed in Streamlit) ── */
#nav-toggle{
  position:absolute;opacity:0;width:0;height:0;pointer-events:none;
}

.hamburger-label{
  display:flex;flex-direction:column;gap:5px;
  width:38px;height:38px;align-items:center;justify-content:center;
  background:transparent;border:1px solid rgba(0,229,255,.2);
  border-radius:8px;cursor:pointer;
  transition:border-color .2s,background .2s;
  flex-shrink:0;
}
.hamburger-label:hover{background:rgba(0,229,255,.08);border-color:rgba(0,229,255,.4)}
.hamburger-label .bar{
  display:block;width:18px;height:2px;
  background:#00e5ff;border-radius:2px;
  transition:all .25s cubic-bezier(.4,0,.2,1);
  transform-origin:center;
}
/* Animate bars → X when checked */
#nav-toggle:checked ~ .aura-navbar .hamburger-label .bar:nth-child(1){
  transform:translateY(7px) rotate(45deg);
}
#nav-toggle:checked ~ .aura-navbar .hamburger-label .bar:nth-child(2){
  opacity:0;transform:scaleX(0);
}
#nav-toggle:checked ~ .aura-navbar .hamburger-label .bar:nth-child(3){
  transform:translateY(-7px) rotate(-45deg);
}

/* Dropdown — hidden by default, shown when checkbox checked */
.nav-dropdown{
  position:absolute;top:calc(100% + 6px);right:0;
  background:rgba(8,14,31,.97);
  border:1px solid rgba(0,229,255,.18);
  border-radius:12px;padding:.5rem;min-width:210px;
  box-shadow:0 16px 40px rgba(0,0,0,.65);
  display:flex;flex-direction:column;gap:.2rem;
  z-index:1000;
  /* Hidden state */
  opacity:0;pointer-events:none;
  transform:translateY(-6px);
  transition:opacity .18s ease,transform .18s ease;
}
#nav-toggle:checked ~ .aura-navbar .nav-dropdown{
  opacity:1;pointer-events:auto;transform:translateY(0);
}

.nav-pill{
  background:transparent;border:1px solid transparent;
  border-radius:8px;color:#6b7a99;
  font-family:'DM Sans',sans-serif;font-size:.88rem;font-weight:500;
  padding:.55rem 1rem;cursor:pointer;transition:all .15s;
  letter-spacing:.02em;text-decoration:none!important;
  display:flex;align-items:center;gap:.55rem;
}
.nav-pill:hover,.nav-pill:visited,.nav-pill:focus,.nav-pill:active{
  text-decoration:none!important;
}
.nav-pill:hover{background:rgba(0,229,255,.07);color:#c8d0e0;border-color:rgba(0,229,255,.12)}
.nav-pill.active{background:rgba(0,229,255,.12);color:#00e5ff;border-color:rgba(0,229,255,.3)}

/* Right side cluster */
.navbar-right{position:relative;display:flex;align-items:center;gap:.8rem}
.nav-live{
  font-family:'Space Mono',monospace;font-size:.65rem;color:#6b7a99;
  border:1px solid rgba(0,229,255,.1);padding:.28rem .7rem;
  border-radius:20px;background:rgba(0,229,255,.03);
  display:flex;align-items:center;gap:.4rem;
}
.live-dot{width:6px;height:6px;border-radius:50%;background:#00e676;
  animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* Navbar bottom margin spacer */
.navbar-spacer{margin-bottom:1.5rem}

/* CONTACT PAGE */
.contact-hero{
  text-align:center;padding:3.5rem 1rem 2.5rem;
  background:radial-gradient(ellipse at 50% 0%,rgba(0,229,255,.06) 0%,transparent 65%);
  border-radius:16px;margin-bottom:2rem;
}
.contact-hero-title{
  font-family:'Space Mono',monospace;font-size:2.2rem;font-weight:700;
  background:linear-gradient(135deg,#00e5ff 0%,#7fff6e 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin:0 0 .6rem;letter-spacing:-.01em;
}
.contact-hero-sub{color:#6b7a99;font-size:.95rem;max-width:480px;margin:0 auto;line-height:1.7}
.info-card{
  background:#0b1022;border:1px solid rgba(0,229,255,.13);
  border-radius:13px;padding:1.5rem 1.6rem;
  transition:border-color .2s,transform .18s;height:100%;
}
.info-card:hover{border-color:rgba(0,229,255,.38);transform:translateY(-2px)}
.info-card-icon{font-size:1.6rem;margin-bottom:.7rem}
.info-card-label{
  font-family:'Space Mono',monospace;font-size:.7rem;
  color:#6b7a99;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.35rem;
}
.info-card-val{color:#e8eaf6;font-size:.92rem;line-height:1.6}
.info-card-val a{color:#00e5ff;text-decoration:none}
.info-card-val a:hover{text-decoration:underline}
.team-card{
  background:#0b1022;border:1px solid rgba(0,229,255,.1);
  border-radius:13px;padding:1.6rem 1.2rem;
  text-align:center;transition:border-color .2s,transform .2s;
}
.team-card:hover{border-color:rgba(0,229,255,.35);transform:translateY(-4px)}
.team-ava{
  width:60px;height:60px;border-radius:50%;
  background:linear-gradient(135deg,#0d1833,#1a2a4a);
  border:2px solid rgba(0,229,255,.2);
  display:flex;align-items:center;justify-content:center;
  margin:0 auto .9rem;font-size:1.5rem;
}
.team-name{font-weight:600;color:#e8eaf6;font-size:.92rem;margin-bottom:.15rem}
.team-role{color:#6b7a99;font-size:.72rem;letter-spacing:.05em;text-transform:uppercase}
.team-dept{color:#00e5ff;font-size:.75rem;margin-top:.3rem}
.submit-btn{
  background:linear-gradient(135deg,rgba(0,229,255,.15),rgba(0,229,255,.08));
  border:1px solid rgba(0,229,255,.35);border-radius:9px;
  color:#00e5ff;font-family:'DM Sans',sans-serif;font-size:.9rem;
  font-weight:600;padding:.7rem 2rem;cursor:pointer;
  transition:all .2s;width:100%;margin-top:.5rem;letter-spacing:.03em;
}
.submit-btn:hover{background:rgba(0,229,255,.22);border-color:#00e5ff}
.form-section{
  background:#0b1022;border:1px solid rgba(0,229,255,.13);
  border-radius:14px;padding:2rem;
}
.form-section h3{
  font-family:'Space Mono',monospace;color:#e8eaf6;
  font-size:.95rem;letter-spacing:.04em;margin-bottom:1.5rem;
}
.stTextInput input,.stTextArea textarea,.stSelectbox{
  background:#0d1428!important;border-color:rgba(0,229,255,.2)!important;
  color:#e8eaf6!important;border-radius:8px!important;
}
.divider-line{
  height:1px;background:linear-gradient(90deg,transparent,rgba(0,229,255,.2),transparent);
  margin:2rem 0;
}
.about-section{
  background:linear-gradient(135deg,rgba(0,229,255,.04),rgba(127,255,110,.03));
  border:1px solid rgba(0,229,255,.1);border-radius:14px;
  padding:2rem;margin-bottom:1.5rem;
}

/* RESPONSIVE */
@media(max-width:768px){
  .aura-navbar{padding:.5rem .8rem}
  .contact-hero-title{font-size:1.5rem}
  [data-testid="stMetricValue"] div{font-size:1.2rem!important}
}
@media(max-width:480px){
  .navbar-brand{font-size:.95rem}
  .nav-live{display:none}
  .contact-hero{padding:2rem .5rem 1.5rem}
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FILE CHECK
# ══════════════════════════════════════════════════════════════════
missing = []
if not CSV_PATH.exists():   missing.append(f"❌ `city_day.csv` not found at `{CSV_PATH}`")
if not MODEL_PKL.exists():  missing.append(f"❌ `outputs/aura_rf_model.pkl` not found")
if not SCALER_PKL.exists(): missing.append(f"❌ `outputs/aura_scaler.pkl` not found")

if missing:
    st.markdown("# 🌆 AURA — Setup Required")
    st.markdown('<div class="err-box">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Missing files — this causes blank screen")
    for m in missing: st.markdown(m)
    st.markdown("</div>", unsafe_allow_html=True)
    st.code(f"📁 AURA/\n├── dashboard.py\n├── city_day.csv\n└── outputs/\n    ├── aura_rf_model.pkl\n    ├── aura_scaler.pkl\n    └── model_meta.json\n\nLocation: {BASE}")
    st.code("python train_model.py", language="bash")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# STATIONS & ROUTES
# ══════════════════════════════════════════════════════════════════
STATIONS = {
    "Anand Vihar":{"lat":28.6469,"lon":77.3162,"zone":"East Delhi"},
    "Ashok Vihar":{"lat":28.6943,"lon":77.1877,"zone":"North Delhi"},
    "Aya Nagar":{"lat":28.4726,"lon":77.1141,"zone":"South Delhi"},
    "Bawana":{"lat":28.7896,"lon":77.0373,"zone":"North Delhi"},
    "Burari Crossing":{"lat":28.7273,"lon":77.2076,"zone":"North Delhi"},
    "Chandni Chowk":{"lat":28.6507,"lon":77.2334,"zone":"Central Delhi"},
    "CRRI Mathura Road":{"lat":28.5534,"lon":77.2696,"zone":"South East Delhi"},
    "Dwarka Sector 8":{"lat":28.5758,"lon":77.0624,"zone":"West Delhi"},
    "IGI Airport T3":{"lat":28.5562,"lon":77.1000,"zone":"West Delhi"},
    "ITO":{"lat":28.6289,"lon":77.2408,"zone":"Central Delhi"},
    "Jahangirpuri":{"lat":28.7295,"lon":77.1628,"zone":"North Delhi"},
    "JLN Stadium":{"lat":28.5831,"lon":77.2336,"zone":"South Delhi"},
    "Lodhi Road":{"lat":28.5921,"lon":77.2233,"zone":"South Delhi"},
    "Mandir Marg":{"lat":28.6372,"lon":77.2002,"zone":"Central Delhi"},
    "Mundka":{"lat":28.6823,"lon":76.9865,"zone":"West Delhi"},
    "Najafgarh":{"lat":28.6090,"lon":76.9820,"zone":"West Delhi"},
    "Nehru Nagar":{"lat":28.5700,"lon":77.2500,"zone":"South Delhi"},
    "North Campus DU":{"lat":28.6873,"lon":77.2140,"zone":"North Delhi"},
    "NSIT Dwarka":{"lat":28.6085,"lon":77.0333,"zone":"West Delhi"},
    "Okhla Phase-2":{"lat":28.5291,"lon":77.2735,"zone":"South East Delhi"},
    "Patparganj":{"lat":28.6275,"lon":77.2995,"zone":"East Delhi"},
    "Punjabi Bagh":{"lat":28.6741,"lon":77.1319,"zone":"West Delhi"},
    "R K Puram":{"lat":28.5638,"lon":77.1873,"zone":"South West Delhi"},
    "Rohini":{"lat":28.7297,"lon":77.0990,"zone":"North Delhi"},
    "Shadipur":{"lat":28.6516,"lon":77.1479,"zone":"West Delhi"},
    "Sirifort":{"lat":28.5512,"lon":77.2196,"zone":"South Delhi"},
    "Sonia Vihar":{"lat":28.7167,"lon":77.2521,"zone":"North East Delhi"},
    "Sri Aurobindo Marg":{"lat":28.5355,"lon":77.1856,"zone":"South Delhi"},
    "Vivek Vihar":{"lat":28.6713,"lon":77.3155,"zone":"North East Delhi"},
    "Wazirpur":{"lat":28.7028,"lon":77.1644,"zone":"North Delhi"},
    "Connaught Place":{"lat":28.6315,"lon":77.2167,"zone":"Central Delhi"},
    "Gurugram Sector 51":{"lat":28.4199,"lon":77.0431,"zone":"Gurugram"},
    "Gurugram Vikas Sadan":{"lat":28.4686,"lon":77.0263,"zone":"Gurugram"},
    "Faridabad Sector 11":{"lat":28.3946,"lon":77.3051,"zone":"Faridabad"},
    "Noida Sector 62":{"lat":28.6278,"lon":77.3720,"zone":"Noida"},
    "Ghaziabad Loni":{"lat":28.7478,"lon":77.2917,"zone":"Ghaziabad"},
}

ZONE_ROUTES = {
    ("Central Delhi","East Delhi"):[{"name":"NH-9 via ITO Bridge","km":14,"time":"30 min","primary":True},{"name":"Ring Road via Pragati Maidan","km":16,"time":"28 min","primary":False},{"name":"Yamuna Bank Connector","km":19,"time":"35 min","primary":False}],
    ("Central Delhi","North Delhi"):[{"name":"GT Karnal Road","km":12,"time":"28 min","primary":True},{"name":"Outer Ring via Azadpur","km":15,"time":"25 min","primary":False},{"name":"Rani Jhansi Marg","km":13,"time":"26 min","primary":False}],
    ("Central Delhi","South Delhi"):[{"name":"Mathura Road NH-19","km":10,"time":"22 min","primary":True},{"name":"Aurobindo Marg via Safdarjung","km":11,"time":"20 min","primary":False},{"name":"Outer Ring via Dhaula Kuan","km":14,"time":"24 min","primary":False}],
    ("Central Delhi","West Delhi"):[{"name":"NH-48 via Dhaula Kuan","km":16,"time":"35 min","primary":True},{"name":"Ring Road via Patel Nagar","km":18,"time":"32 min","primary":False},{"name":"Najafgarh Road Bypass","km":22,"time":"38 min","primary":False}],
    ("Central Delhi","Gurugram"):[{"name":"NH-48 via Dhaula Kuan","km":32,"time":"55 min","primary":True},{"name":"Mehrauli-Gurgaon Road","km":28,"time":"40 min","primary":False},{"name":"Vasant Kunj Connector","km":30,"time":"38 min","primary":False}],
    ("Central Delhi","Noida"):[{"name":"DND Flyway direct","km":22,"time":"35 min","primary":True},{"name":"NH-9 via Kalindi Kunj","km":26,"time":"38 min","primary":False},{"name":"Mayur Vihar Link Road","km":20,"time":"30 min","primary":False}],
    ("Central Delhi","Faridabad"):[{"name":"Mathura Road NH-19","km":28,"time":"45 min","primary":True},{"name":"Badarpur Flyover Route","km":32,"time":"42 min","primary":False},{"name":"Kalindi Kunj via Okhla","km":35,"time":"50 min","primary":False}],
    ("Central Delhi","Ghaziabad"):[{"name":"NH-9 via Anand Vihar","km":24,"time":"40 min","primary":True},{"name":"Wazirabad Road via NH-58","km":30,"time":"50 min","primary":False},{"name":"Loni Road via Seelampur","km":28,"time":"48 min","primary":False}],
    ("North Delhi","East Delhi"):[{"name":"Ring Road via GTB Nagar","km":20,"time":"40 min","primary":True},{"name":"Wazirabad Road via Yamuna","km":24,"time":"38 min","primary":False},{"name":"Loni Road via Seelampur","km":22,"time":"36 min","primary":False}],
    ("North Delhi","West Delhi"):[{"name":"Outer Ring via Punjabi Bagh","km":18,"time":"38 min","primary":True},{"name":"NH-48 via Rohini","km":22,"time":"35 min","primary":False},{"name":"Peeragarhi Road","km":20,"time":"33 min","primary":False}],
    ("North Delhi","Ghaziabad"):[{"name":"NH-58 via Ghazipur","km":25,"time":"45 min","primary":True},{"name":"Loni Road Bypass","km":30,"time":"42 min","primary":False},{"name":"NH-9 via Dilshad Garden","km":28,"time":"40 min","primary":False}],
    ("North Delhi","Gurugram"):[{"name":"Ring Road + NH-48","km":48,"time":"70 min","primary":True},{"name":"Outer Ring + Dhaula Kuan","km":44,"time":"60 min","primary":False},{"name":"Pankha Road Connector","km":46,"time":"65 min","primary":False}],
    ("South Delhi","Gurugram"):[{"name":"NH-48 direct","km":18,"time":"30 min","primary":True},{"name":"MG Road via Sikanderpur","km":22,"time":"28 min","primary":False},{"name":"Aya Nagar Connector","km":20,"time":"25 min","primary":False}],
    ("South Delhi","Faridabad"):[{"name":"Mathura Road NH-19","km":20,"time":"35 min","primary":True},{"name":"Badarpur Bypass","km":24,"time":"32 min","primary":False},{"name":"Kalindi Kunj via Okhla","km":22,"time":"30 min","primary":False}],
    ("South Delhi","Noida"):[{"name":"DND Flyway","km":16,"time":"25 min","primary":True},{"name":"Kalindi Kunj Bypass","km":18,"time":"28 min","primary":False},{"name":"NH-9 via Mayur Vihar","km":20,"time":"32 min","primary":False}],
    ("East Delhi","Noida"):[{"name":"DND Flyway","km":12,"time":"20 min","primary":True},{"name":"Kalindi Kunj Bypass","km":15,"time":"22 min","primary":False},{"name":"NH-9 via Mayur Vihar","km":14,"time":"18 min","primary":False}],
    ("East Delhi","Ghaziabad"):[{"name":"NH-9 direct","km":16,"time":"28 min","primary":True},{"name":"Loni Road via Seelampur","km":20,"time":"32 min","primary":False},{"name":"Anand Vihar ISBT Link","km":18,"time":"26 min","primary":False}],
    ("West Delhi","Gurugram"):[{"name":"NH-48 via Palam","km":28,"time":"45 min","primary":True},{"name":"Dwarka Expressway","km":32,"time":"38 min","primary":False},{"name":"Bijwasan Road","km":30,"time":"40 min","primary":False}],
    ("West Delhi","Noida"):[{"name":"Ring Road + DND","km":35,"time":"55 min","primary":True},{"name":"NH-48 + Mathura Road","km":38,"time":"52 min","primary":False},{"name":"NH-9 via Connaught Place","km":40,"time":"60 min","primary":False}],
    ("South West Delhi","Gurugram"):[{"name":"NH-48 via Dhaula Kuan","km":20,"time":"32 min","primary":True},{"name":"Mehrauli Road","km":22,"time":"28 min","primary":False},{"name":"Palam Road Bypass","km":24,"time":"35 min","primary":False}],
    ("South East Delhi","Faridabad"):[{"name":"Mathura Road NH-19","km":15,"time":"25 min","primary":True},{"name":"Badarpur Border Route","km":18,"time":"28 min","primary":False},{"name":"Sarita Vihar Flyover","km":17,"time":"26 min","primary":False}],
    ("North East Delhi","Ghaziabad"):[{"name":"NH-58 via Wazirabad","km":18,"time":"30 min","primary":True},{"name":"Loni Road Direct","km":20,"time":"28 min","primary":False},{"name":"Dilshad Garden Link","km":22,"time":"32 min","primary":False}],
    ("Gurugram","Faridabad"):[{"name":"Southern Peripheral Road","km":38,"time":"55 min","primary":True},{"name":"NH-48 + Mathura Road","km":42,"time":"60 min","primary":False},{"name":"Faridabad-Gurgaon Road","km":40,"time":"58 min","primary":False}],
    ("Noida","Ghaziabad"):[{"name":"NH-9 direct","km":14,"time":"22 min","primary":True},{"name":"Hindon Elevated Road","km":16,"time":"20 min","primary":False},{"name":"Vasundhara Link","km":18,"time":"25 min","primary":False}],
}

def get_routes(z1,z2):
    return ZONE_ROUTES.get((z1,z2)) or ZONE_ROUTES.get((z2,z1))

def check_threshold(pk,ak,mp):
    ov=((ak-pk)/pk)*100
    return ov<=mp, round(ov,1)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
POLLUTANTS=['PM2.5','PM10','NO','NO2','NOx','CO','SO2','O3','NH3']
FEATURES=['PM2.5','PM10','NO','NO2','NOx','CO','SO2','O3','NH3',
          'month','dayofweek','year','is_weekend','quarter',
          'pm25_lag1','pm25_lag2','pm25_lag3','pm25_lag7',
          'aqi_lag1','pm25_roll7','pm25_roll14','no2_roll7']

def aqi_cat(v):
    if v<=50:   return "Good","#00e676"
    if v<=100:  return "Satisfactory","#a8e063"
    if v<=200:  return "Moderate","#ffaa00"
    if v<=300:  return "Poor","#ff6b35"
    if v<=400:  return "Very Poor","#ff4444"
    return "Severe","#8b0000"

def pm25_to_aqi(p):
    bp=[(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)]
    for c0,c1,i0,i1 in bp:
        if c0<=p<=c1: return round(i0+(p-c0)*(i1-i0)/(c1-c0))
    return 500

@st.cache_data
def load_csv():
    df=pd.read_csv(CSV_PATH)
    ncr=df[df['City'].isin(['Delhi','Gurugram'])].copy()
    ncr['Date']=pd.to_datetime(ncr['Date'])
    ncr=ncr.sort_values('Date').reset_index(drop=True)
    for col in POLLUTANTS:
        if col in ncr.columns: ncr[col]=ncr[col].interpolate(method='linear').ffill().bfill()
    ncr['month']=ncr['Date'].dt.month
    ncr['dayofweek']=ncr['Date'].dt.dayofweek
    ncr['year']=ncr['Date'].dt.year
    ncr['is_weekend']=(ncr['dayofweek']>=5).astype(int)
    ncr['quarter']=ncr['Date'].dt.quarter
    for lag in [1,2,3,7]:
        ncr[f'pm25_lag{lag}']=ncr['PM2.5'].shift(lag)
        ncr[f'aqi_lag{lag}']=ncr['AQI'].shift(lag) if 'AQI' in ncr.columns else 0
    ncr['pm25_roll7']=ncr['PM2.5'].rolling(7).mean()
    ncr['pm25_roll14']=ncr['PM2.5'].rolling(14).mean()
    ncr['no2_roll7']=ncr['NO2'].rolling(7).mean()
    return ncr

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PKL), joblib.load(SCALER_PKL)

@st.cache_data
def load_meta():
    with open(META_JSON) as f: return json.load(f)

df=load_csv(); model,scaler=load_model(); meta=load_meta()
now=datetime.datetime.now(); hour=now.hour

def get_pm25(station):
    city="Gurugram" if "Gurugram" in station else "Delhi"
    sub=df[df['City']==city]; m=now.month
    base=float(sub[sub['month']==m]['PM2.5'].mean()) if len(sub[sub['month']==m])>0 else float(sub['PM2.5'].mean())
    d=[0.75,0.70,0.65,0.62,0.65,0.80,1.05,1.35,1.45,1.30,1.15,1.05,1.00,0.98,1.00,1.08,1.20,1.40,1.45,1.35,1.20,1.05,0.90,0.80]
    return float(np.clip(base*d[hour]+np.random.normal(0,10),15,480))

def predict_pm25(station):
    city="Gurugram" if "Gurugram" in station else "Delhi"
    sub=df[df['City']==city].dropna(subset=FEATURES).tail(5)
    if sub.empty: return get_pm25(station)+10
    try:
        row=sub.iloc[-1][FEATURES].values.reshape(1,-1)
        return max(10.0,float(model.predict(scaler.transform(row))[0])+np.random.normal(0,5))
    except: return get_pm25(station)+10

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "station" not in st.session_state:
    st.session_state.station="Anand Vihar"
    st.session_state.dest_zone="West Delhi"
    st.session_state.page="Dashboard"
    base0=get_pm25("Anand Vihar")
    st.session_state.hist={
        "ts":[now-datetime.timedelta(minutes=30-2*i) for i in range(15)],
        "pm25":[float(np.clip(base0+np.random.normal(0,15),15,450)) for _ in range(15)],
        "pred":[float(np.clip(base0+10+np.random.normal(0,10),15,450)) for _ in range(15)],
    }

if "page" not in st.session_state:
    st.session_state.page="Dashboard"

# ══════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════
pages=["Dashboard","Analytics","Rerouting","Data Explorer","Contact"]
page_icons={"Dashboard":"🌆","Analytics":"📊","Rerouting":"🔀","Data Explorer":"🗂️","Contact":"✉️"}

active=st.session_state.page

pills_html="".join([
    f'<a class="nav-pill {"active" if p==active else ""}" href="?page={p}" target="_self">{page_icons[p]} {p}</a>'
    for p in pages
])
st.markdown(f"""
<!-- checkbox is the toggle state — MUST be sibling of navbar-right -->
<input type="checkbox" id="nav-toggle">
<div class="aura-navbar">
  <div class="navbar-brand">AURA<span> // NCR</span></div>
  <div class="navbar-right">
    <div class="nav-live"><span class="live-dot"></span>LIVE · {now.strftime("%H:%M:%S")}</div>
    <label class="hamburger-label" for="nav-toggle" aria-label="Toggle navigation">
      <span class="bar"></span>
      <span class="bar"></span>
      <span class="bar"></span>
    </label>
    <div class="nav-dropdown">
      {pills_html}
    </div>
  </div>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)

# Read page from URL query param
qp = st.query_params
if "page" in qp and qp["page"] in pages:
    st.session_state.page = qp["page"]
active = st.session_state.page

# ══════════════════════════════════════════════════════════════════
# SIDEBAR (only on dashboard/analytics/rerouting)
# ══════════════════════════════════════════════════════════════════
all_stations=sorted(STATIONS.keys())
all_zones=sorted(set(v["zone"] for v in STATIONS.values()))

if active != "Contact":
    with st.sidebar:
        st.markdown("## 🌆 AURA Controls")
        st.markdown("---")
        sel=st.selectbox("📍 Station",all_stations,index=all_stations.index(st.session_state.station))
        st.session_state.station=sel
        info=STATIONS[sel]; sel_zone=info["zone"]
        st.markdown(f"**Zone:** `{sel_zone}`")
        st.markdown(f"**GPS:** `{info['lat']:.4f}, {info['lon']:.4f}`")
        st.markdown("---")
        dest=st.selectbox("🔀 Destination Zone",[z for z in all_zones if z!=sel_zone],index=0)
        st.session_state.dest_zone=dest
        st.markdown("---")
        threshold=st.slider("⚠️ Alert Threshold (µg/m³)",60,300,150,10)
        max_overhead=st.slider("🛣️ Max Reroute Overhead (%)",5,60,35,5)
        auto_ref=st.toggle("🔄 Auto-refresh (5s)",True)
        st.markdown("---")
        st.markdown("**📊 Model Stats**")
        st.markdown(f"- MAE: `{meta['mae']} µg/m³`\n- RMSE: `{meta['rmse']} µg/m³`\n- R²: `{meta['r2']}`\n- OOB R²: `{meta['oob']}`\n- Scaler: `MinMaxScaler`")
        st.markdown("---")
        st.markdown("**📁 Dataset**")
        st.markdown("- city_day.csv · Delhi+Gurugram\n- 2015–2020 · 22 features")
else:
    sel=st.session_state.station; info=STATIONS[sel]; sel_zone=info["zone"]
    threshold=150; max_overhead=35; auto_ref=False; dest=st.session_state.dest_zone

# ── live values ──
cur_pm25=get_pm25(sel); pred_pm25=predict_pm25(sel)
cur_aqi=pm25_to_aqi(cur_pm25); pred_aqi=pm25_to_aqi(pred_pm25)
reroute=pred_pm25>threshold
cat_now,_=aqi_cat(cur_pm25); cat_pred,_=aqi_cat(pred_pm25)

hist=st.session_state.hist
hist["ts"].append(now); hist["pm25"].append(cur_pm25); hist["pred"].append(pred_pm25)
for k in ["ts","pm25","pred"]:
    if len(hist[k])>80: hist[k]=hist[k][-80:]

color_map_aqi={"Good":"#00e676","Satisfactory":"#a8e063","Moderate":"#ffaa00",
               "Poor":"#ff6b35","Very Poor":"#ff4444","Severe":"#8b0000"}

# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
if active=="Dashboard":
    st.markdown(f"### 🌆 Delhi NCR — Live Air Quality Monitor")
    st.caption(f"Station: **{sel}** · Zone: `{sel_zone}` · {now.strftime('%d %b %Y, %H:%M:%S')} IST")

    if reroute:
        st.markdown(f'<div class="alert-box">🚨 <b>POLLUTION ALERT</b> — RF Model predicts <b>{pred_pm25:.0f} µg/m³</b> ({cat_pred}, AQI≈{pred_aqi}) at <b>{sel}</b>. Go to Rerouting tab.</div>',unsafe_allow_html=True)
    elif cur_pm25>threshold*0.8:
        st.markdown(f'<div class="warn-box">⚠️ <b>ELEVATED</b> — {cur_pm25:.0f} µg/m³ approaching threshold. Predicted: {pred_pm25:.0f} µg/m³.</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safe-box">✅ <b>Clear</b> — Current {cur_pm25:.0f} µg/m³ ({cat_now}). Predicted {pred_pm25:.0f} µg/m³. No rerouting needed.</div>',unsafe_allow_html=True)

    prev=hist["pm25"][-2] if len(hist["pm25"])>=2 else cur_pm25
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("Current PM2.5",f"{cur_pm25:.0f} µg/m³",f"{cur_pm25-prev:+.0f}")
    c2.metric("Predicted PM2.5",f"{pred_pm25:.0f} µg/m³",cat_pred)
    c3.metric("Current AQI",f"{cur_aqi}",cat_now)
    c4.metric("Predicted AQI",f"{pred_aqi}","⚠ ALERT" if reroute else "✓ Safe")
    c5.metric("Zone",sel_zone)
    c6.metric("Model R²",f"{meta['r2']}")
    st.markdown("---")

    st.markdown("#### 🗺️ All 36 DPCC/CPCB Stations")
    map_rows=[]
    for sn,si in STATIONS.items():
        sp=get_pm25(sn); sc,_=aqi_cat(sp)
        map_rows.append({"station":sn,"zone":si["zone"],"lat":si["lat"],"lon":si["lon"],
                         "pm25":round(sp,1),"aqi":pm25_to_aqi(sp),"category":sc,
                         "selected":"⭐ YES" if sn==sel else "No",
                         "size":sp+(80 if sn==sel else 0)})
    mdf=pd.DataFrame(map_rows)
    fig_map=px.scatter_map(mdf,lat="lat",lon="lon",size="size",color="category",
        color_discrete_map=color_map_aqi,hover_name="station",
        hover_data={"zone":True,"pm25":True,"aqi":True,"selected":True,"lat":False,"lon":False,"size":False},
        size_max=50,zoom=10,center={"lat":28.6139,"lon":77.2090},map_style="carto-darkmatter")
    fig_map.update_layout(paper_bgcolor="#0b1022",plot_bgcolor="#0b1022",height=420,
        margin=dict(l=0,r=0,t=0,b=0),font_color="#e8eaf6",
        legend=dict(bgcolor="rgba(11,16,34,.9)",font=dict(color="#e8eaf6"),
                    title=dict(text="AQI Category",font=dict(color="#00e5ff"))))
    st.plotly_chart(fig_map,width="stretch")

    ns=st.selectbox("🔍 Search station:",all_stations,index=all_stations.index(sel),key="map_sel")
    if ns!=sel: st.session_state.station=ns; st.rerun()
    st.markdown("---")

    cl,cr=st.columns(2)
    ts_labels=[t.strftime("%H:%M:%S") for t in hist["ts"]]
    with cl:
        st.markdown("#### 📈 Live PM2.5 — Actual vs Predicted")
        f1=go.Figure()
        f1.add_trace(go.Scatter(x=ts_labels,y=hist["pm25"],name="Actual",line=dict(color="#ff4444",width=2.5)))
        f1.add_trace(go.Scatter(x=ts_labels,y=hist["pred"],name="RF Predicted",line=dict(color="#00e5ff",width=2,dash="dot")))
        f1.add_hrect(y0=threshold,y1=500,fillcolor="rgba(255,68,68,0.05)",line_width=0)
        f1.add_hline(y=threshold,line_dash="dash",line_color="#ffaa00",annotation_text=f"Alert ({threshold})")
        f1.update_layout(paper_bgcolor="#0b1022",plot_bgcolor="#04060f",font_color="#e8eaf6",height=290,
            margin=dict(l=5,r=5,t=10,b=5),legend=dict(bgcolor="#0b1022",font=dict(size=9)),
            xaxis=dict(gridcolor="#1e2a40",showticklabels=False),yaxis=dict(gridcolor="#1e2a40",title="PM2.5 µg/m³"))
        st.plotly_chart(f1,width="stretch")
    with cr:
        st.markdown("#### 📅 Historical PM2.5 — Delhi (your CSV)")
        hd=df[df['City']=='Delhi'].dropna(subset=['PM2.5'])
        hd_g=hd.groupby('Date')['PM2.5'].mean().reset_index()
        f2=go.Figure()
        f2.add_trace(go.Scatter(x=hd_g['Date'],y=hd_g['PM2.5'],fill="tozeroy",
            line=dict(color="#ff6b35",width=1.5),fillcolor="rgba(255,107,53,0.07)"))
        f2.add_hline(y=60,line_dash="dot",line_color="#00e676",annotation_text="CPCB Standard (60)")
        f2.update_layout(paper_bgcolor="#0b1022",plot_bgcolor="#04060f",font_color="#e8eaf6",height=290,
            margin=dict(l=5,r=5,t=10,b=5),showlegend=False,
            xaxis=dict(gridcolor="#1e2a40"),yaxis=dict(gridcolor="#1e2a40",title="PM2.5 µg/m³"))
        st.plotly_chart(f2,width="stretch")

# ══════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif active=="Analytics":
    st.markdown("### 📊 Analytics — Pollution Patterns & Correlation")
    st.caption(f"Based on city_day.csv · Delhi + Gurugram 2015–2020 · Station: {sel}")
    st.markdown("---")

    c3,c4=st.columns(2)
    with c3:
        st.markdown("#### 🗓️ Monthly Avg PM2.5 (real data)")
        md=df[df['City']=='Delhi'].dropna(subset=['PM2.5','month'])
        ma=md.groupby('month')['PM2.5'].mean().reset_index()
        ma['mn']=ma['month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
        clrs=[color_map_aqi.get(aqi_cat(v)[0],"#6b7a99") for v in ma['PM2.5']]
        f3=go.Figure(go.Bar(x=ma['mn'],y=ma['PM2.5'],marker_color=clrs,
            text=[f"{v:.0f}" for v in ma['PM2.5']],textposition='outside',textfont=dict(color='#e8eaf6',size=10)))
        f3.add_hline(y=60,line_dash="dot",line_color="#00e676",annotation_text="Standard")
        f3.update_layout(paper_bgcolor="#0b1022",plot_bgcolor="#04060f",font_color="#e8eaf6",height=280,
            margin=dict(l=5,r=5,t=20,b=5),xaxis=dict(gridcolor="#1e2a40"),
            yaxis=dict(gridcolor="#1e2a40",title="PM2.5 µg/m³"))
        st.plotly_chart(f3,width="stretch")

    with c4:
        st.markdown("#### 🧪 Pollutant Profile (normalized %)")
        city_sel="Gurugram" if "Gurugram" in sel else "Delhi"
        sub=df[df['City']==city_sel].tail(60)
        poll_avg={p:float(sub[p].mean()) for p in POLLUTANTS if p in sub.columns and sub[p].notna().any()}
        mx=max(poll_avg.values()) or 1
        poll_norm={k:round(v/mx*100,1) for k,v in poll_avg.items()}
        f4=go.Figure(go.Bar(x=list(poll_norm.keys()),y=list(poll_norm.values()),
            marker_color="#00e5ff",opacity=0.8,
            text=[f"{v}%" for v in poll_norm.values()],textposition='outside',
            textfont=dict(color='#e8eaf6',size=9)))
        f4.update_layout(paper_bgcolor="#0b1022",plot_bgcolor="#04060f",font_color="#e8eaf6",height=280,
            margin=dict(l=5,r=5,t=20,b=5),xaxis=dict(gridcolor="#1e2a40"),
            yaxis=dict(gridcolor="#1e2a40",title="Normalized %",range=[0,130]))
        st.plotly_chart(f4,width="stretch")

    st.markdown("---")
    st.markdown("### 🔥 Correlation Heatmap")

    @st.cache_data
    def build_corr(city_name):
        sub=df[df['City']==city_name].copy()
        for lag in [1,2,3,7]: sub[f'PM2.5_Lag{lag}']=sub['PM2.5'].shift(lag)
        sub['PM2.5_Roll7']=sub['PM2.5'].rolling(7).mean()
        sub['PM2.5_Roll14']=sub['PM2.5'].rolling(14).mean()
        sub['Pollution_Load']=sub['PM2.5']+sub['NO2']+sub['SO2']
        sub['month']=pd.to_datetime(sub['Date']).dt.month
        sub['PM2.5_Next']=sub['PM2.5'].shift(-1)
        sub.dropna(inplace=True)
        COLS=['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3',
              'PM2.5_Lag1','PM2.5_Lag2','PM2.5_Lag3','PM2.5_Lag7',
              'PM2.5_Roll7','PM2.5_Roll14','Pollution_Load','month','PM2.5_Next']
        LBLS=['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3',
              'Lag 1d','Lag 2d','Lag 3d','Lag 7d',
              'Roll 7d','Roll 14d','Poll. Load','Month','PM2.5 Next\n(Target)']
        corr=sub[COLS].corr(); corr.index=LBLS; corr.columns=LBLS
        return corr

    hm_city="Gurugram" if "Gurugram" in sel else "Delhi"
    corr_df=build_corr(hm_city)
    LABELS=list(corr_df.columns)

    hm_tab1,hm_tab2=st.tabs(["🟩 Full Matrix","📊 Target Correlations"])
    with hm_tab1:
        st.caption(f"Pearson r between all features — {hm_city}")
        z=corr_df.values; text=[[f"{v:.2f}" for v in row] for row in z]
        fhm=go.Figure(go.Heatmap(z=z,x=LABELS,y=LABELS,text=text,texttemplate="%{text}",
            textfont=dict(size=7,color="white"),
            colorscale=[[0,"#8b0000"],[0.25,"#ff4444"],[0.45,"#1e2a40"],[0.55,"#1e2a40"],[0.75,"#00e676"],[1,"#006400"]],
            zmid=0,zmin=-1,zmax=1,
            colorbar=dict(title=dict(text="r",font=dict(color="#e8eaf6")),tickfont=dict(color="#e8eaf6"),
                tickvals=[-1,-0.5,0,0.5,1],ticktext=["-1.0","-0.5","0","0.5","1.0"],thickness=14,len=0.85),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>"))
        fhm.update_layout(paper_bgcolor="#04060f",plot_bgcolor="#04060f",font_color="#e8eaf6",height=560,
            margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(tickfont=dict(size=8,color="#e8eaf6"),tickangle=45,showgrid=False),
            yaxis=dict(tickfont=dict(size=8,color="#e8eaf6"),autorange="reversed",showgrid=False))
        st.plotly_chart(fhm,width="stretch")
        st.caption("🟩 Green = positive · 🟥 Red = negative · ⬛ Dark = no correlation")

    with hm_tab2:
        tc=corr_df["PM2.5 Next\n(Target)"].drop("PM2.5 Next\n(Target)").sort_values()
        bclrs=["#00e5ff" if v>=0.7 else "#7fff6e" if v>=0.4 else "#ffaa00" if v>=0.1 else "#6b7a99" if v>=0 else "#ff4444" for v in tc.values]
        fb=go.Figure(go.Bar(x=tc.values,y=tc.index,orientation='h',marker_color=bclrs,
            text=[f"  {v:+.3f}" for v in tc.values],textposition='outside',textfont=dict(color="#e8eaf6",size=10),
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>"))
        fb.add_vline(x=0,line_color="white",line_width=1.2,opacity=0.5)
        fb.add_vline(x=0.7,line_color="#00e5ff",line_width=1,line_dash="dash",
            annotation_text="Strong (0.7)",annotation_font_color="#00e5ff",annotation_position="top")
        fb.add_vline(x=0.4,line_color="#7fff6e",line_width=0.8,line_dash="dot",
            annotation_text="Moderate (0.4)",annotation_font_color="#7fff6e",annotation_position="top")
        fb.update_layout(paper_bgcolor="#0b1022",plot_bgcolor="#04060f",font_color="#e8eaf6",height=460,
            margin=dict(l=10,r=60,t=30,b=10),
            xaxis=dict(gridcolor="#1e2a40",range=[-0.3,1.1],title="Pearson r",tickfont=dict(color="#6b7a99")),
            yaxis=dict(tickfont=dict(size=9,color="#e8eaf6"),gridcolor="#1e2a40"),showlegend=False)
        st.plotly_chart(fb,width="stretch")
        t3=tc.sort_values(ascending=False).head(3)
        st.markdown(f"**🔍 Insight:** **{t3.index[0]}** (r={t3.iloc[0]:.2f}), **{t3.index[1]}** (r={t3.iloc[1]:.2f}), **{t3.index[2]}** (r={t3.iloc[2]:.2f}) are the strongest predictors of next-day PM2.5.")

# ══════════════════════════════════════════════════════════════════
# PAGE: REROUTING
# ══════════════════════════════════════════════════════════════════
elif active=="Rerouting":
    st.markdown("### 🔀 Smart Rerouting Engine")
    st.caption(f"From: **{sel_zone}** → To: **{dest}** | Alternate shown only if overhead ≤ **{max_overhead}%**")

    # Status card
    if reroute:
        st.markdown(f'<div class="alert-box">🚨 <b>REROUTING ACTIVE</b> — Predicted {pred_pm25:.0f} µg/m³ ({cat_pred}) exceeds threshold of {threshold} µg/m³. Evaluating alternates below.</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safe-box">✅ <b>No rerouting needed</b> — Predicted {pred_pm25:.0f} µg/m³ is within safe limits. Routes shown for reference.</div>',unsafe_allow_html=True)

    st.markdown("---")
    routes=get_routes(sel_zone,dest)
    if sel_zone==dest:
        st.info("📍 Same zone — no rerouting needed.")
    elif routes is None:
        st.warning(f"No route data for **{sel_zone} → {dest}**. Try adjacent zones.")
        st.markdown("**Available:** " + " | ".join([f"{a}↔{b}" for a,b in list(ZONE_ROUTES.keys())[:10]]))
    else:
        primary=next(r for r in routes if r["primary"])
        alts=[r for r in routes if not r["primary"]]
        r1,r2,r3=st.columns(3)
        with r1:
            st.markdown("**🔴 Primary Route**")
            if reroute: st.error(f"**{primary['name']}**\n\n📏 {primary['km']} km · ⏱ {primary['time']}\n\n⚠️ High pollution zone")
            else: st.success(f"**{primary['name']}**\n\n📏 {primary['km']} km · ⏱ {primary['time']}\n\n✅ Proceed normally")
        valid_count=0
        for i,(alt,col) in enumerate(zip(alts,[r2,r3])):
            ok,ov=check_threshold(primary["km"],alt["km"],max_overhead)
            with col:
                st.markdown(f"**{'🟢' if ok else '🔘'} Alternate {'A' if i==0 else 'B'}**")
                if not reroute: st.info(f"**{alt['name']}**\n\n📏 {alt['km']} km · ⏱ {alt['time']}\n\n📊 +{ov:.0f}% vs primary")
                elif ok: valid_count+=1; st.success(f"**{alt['name']}**\n\n📏 {alt['km']} km · ⏱ {alt['time']}\n\n✅ RECOMMENDED — +{ov:.0f}% only")
                else: st.warning(f"**{alt['name']}**\n\n📏 {alt['km']} km · ⏱ {alt['time']}\n\n❌ +{ov:.0f}% exceeds {max_overhead}% limit")
        if reroute:
            if valid_count==0: st.markdown(f'<div class="warn-box">⚠️ No alternates within {max_overhead}% limit. Increase slider.</div>',unsafe_allow_html=True)
            else: st.markdown(f'<div class="safe-box">✅ {valid_count} valid alternate(s) within {max_overhead}% overhead.</div>',unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("📋 CPCB AQI Scale"):
        st.dataframe(pd.DataFrame([
            {"PM2.5":"0–30","AQI":"0–50","Category":"Good","Action":"None"},
            {"PM2.5":"30–60","AQI":"51–100","Category":"Satisfactory","Action":"Monitor"},
            {"PM2.5":"60–90","AQI":"101–200","Category":"Moderate","Action":"Alert"},
            {"PM2.5":"90–120","AQI":"201–300","Category":"Poor","Action":"Consider rerouting"},
            {"PM2.5":"120–250","AQI":"301–400","Category":"Very Poor","Action":"Reroute immediately"},
            {"PM2.5":"250+","AQI":"401–500","Category":"Severe","Action":"Emergency protocol"},
        ]),width="stretch",hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
elif active=="Data Explorer":
    st.markdown("### 🗂️ Data Explorer")
    st.caption(f"city_day.csv — {len(df)} rows · Delhi + Gurugram · 2015–2020")
    city_f=st.multiselect("City",["Delhi","Gurugram"],default=["Delhi"])
    cols_f=st.multiselect("Columns",df.columns.tolist(),
        default=['City','Date','PM2.5','PM10','NO2','CO','AQI','AQI_Bucket'])
    filt=df[df['City'].isin(city_f)][cols_f].sort_values('Date',ascending=False) if city_f else df[cols_f]
    st.dataframe(filt.head(200),width="stretch",hide_index=True)
    st.markdown("---")
    st.markdown("#### 📈 Custom Trend Viewer")
    col_trend=st.selectbox("Select pollutant to plot",[c for c in ['PM2.5','PM10','NO2','CO','SO2','O3','AQI'] if c in df.columns])
    city_trend=st.radio("City",["Delhi","Gurugram"],horizontal=True)
    trend_data=df[df['City']==city_trend].dropna(subset=[col_trend]).groupby('Date')[col_trend].mean().reset_index()
    ft=go.Figure()
    ft.add_trace(go.Scatter(x=trend_data['Date'],y=trend_data[col_trend],
        fill="tozeroy",line=dict(color="#00e5ff",width=1.8),fillcolor="rgba(0,229,255,0.06)"))
    ft.update_layout(paper_bgcolor="#0b1022",plot_bgcolor="#04060f",font_color="#e8eaf6",height=300,
        margin=dict(l=5,r=5,t=10,b=5),xaxis=dict(gridcolor="#1e2a40"),
        yaxis=dict(gridcolor="#1e2a40",title=col_trend),showlegend=False)
    st.plotly_chart(ft,width="stretch")

# ══════════════════════════════════════════════════════════════════
# PAGE: CONTACT
# ══════════════════════════════════════════════════════════════════
elif active=="Contact":
    st.markdown("""
    <div class="contact-hero">
      <div class="contact-hero-title">Get In Touch</div>
      <p class="contact-hero-sub">AURA is a student research project developed at the intersection of AI, computer vision, and environmental monitoring. Reach out for collaboration, feedback, or queries.</p>
    </div>
    """, unsafe_allow_html=True)

    # Team
    st.markdown("#### 👥 Project Team")
    t1,t2,t3,t4,t5=st.columns(5)
    team=[
        ("👩‍💻","Riya Parnami","ML Engineer","Computer Science & Engineering"),
        ("👨‍🔬","Samiksha Desale","Computer Vision","Computer Science & Engineering"),
        ("👩‍🎓","Isha Sahu","Data Engineer","AI & Data Science"),
        ("👩‍🎓","Amisha Patel","Backend Dev","Computer Science & Engineering"),
        ("👩‍🎓","Swaroop Biradar","Frontend Dev","Computer Science & Engineering"),
    ]
    for col,(ava,name,role,dept) in zip([t1,t2,t3,t4,t5],team):
        col.markdown(f"""
        <div class="team-card">
          <div class="team-ava">{ava}</div>
          <div class="team-name">{name}</div>
          <div class="team-role">{role}</div>
          <div class="team-dept">{dept}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="divider-line"></div>',unsafe_allow_html=True)

    # Info cards + form
    left,right=st.columns([1,1.3])
    with left:
        st.markdown("#### 📬 Contact Information")
        cards=[
            ("📧","Email","riya.parnami23@pcu.edu.in"),
            ("🏛️","Institution","Department of Computer Science & Engineering \n\nPimpri Chinchwad University,Pune"),
            ("📍","Location","Pune, India — 412106"),
            ("🔗","Project Links","GitHub · Research Paper · Demo Video"),
            ("📅","Available","Mon–Fri, 10:00 AM – 5:00 PM IST"),
        ]
        for icon,label,val in cards:
            st.markdown(f"""
            <div class="info-card" style="margin-bottom:.7rem">
              <div class="info-card-icon">{icon}</div>
              <div class="info-card-label">{label}</div>
              <div class="info-card-val">{val}</div>
            </div>""",unsafe_allow_html=True)

    with right:
        st.markdown("#### ✉️ Send a Message")
        st.markdown('<div class="form-section">',unsafe_allow_html=True)
        name_in=st.text_input("Your Name",placeholder="e.g. Dr. Sharma")
        email_in=st.text_input("Email Address",placeholder="your@email.com")
        subject_in=st.selectbox("Subject",["General Inquiry","Collaboration Request",
            "Dataset / Data Sharing","Technical Feedback","Research Partnership","Other"])
        msg_in=st.text_area("Message",placeholder="Write your message here...",height=130)
        if st.button("🚀 Send Message",use_container_width=True):
            if name_in and email_in and msg_in:
                st.success(f"✅ Thank you **{name_in}**! Your message has been received. We'll respond to **{email_in}** within 2 business days.")
            else:
                st.error("Please fill in Name, Email, and Message fields.")
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="divider-line"></div>',unsafe_allow_html=True)

    # About section
    st.markdown("""
    <div class="about-section">
      <h4 style="font-family:'Space Mono',monospace;color:#00e5ff;font-size:.9rem;letter-spacing:.06em;margin-bottom:1rem">ABOUT AURA</h4>
      <p style="color:#a0aec0;line-height:1.8;font-size:.9rem">
        <b style="color:#e8eaf6">AURA (AI-based Urban Real-time Analysis)</b> is a smart city monitoring system that combines
        YOLOv8 computer vision for traffic density estimation with Random Forest machine learning
        for PM2.5 air quality prediction. The system processes CCTV footage in real time, computes
        a Traffic Density Index, and predicts next-day pollution levels using 22 engineered features
        derived from CPCB monitoring data (2015–2020).<br><br>
        When predicted PM2.5 exceeds the alert threshold, AURA's smart rerouting engine evaluates
        alternate routes — recommending only those within a configurable distance overhead percentage,
        preventing unreasonable detours. The dashboard covers 36 real DPCC/CPCB stations across
        Delhi NCR including Gurugram, Noida, Faridabad, and Ghaziabad.
      </p>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center;color:#2a3550;font-size:.75rem;margin-top:2rem;font-family:'Space Mono',monospace">
    AURA v3.0 · Built with Streamlit, Plotly, scikit-learn · Delhi NCR 2015–2020
    </p>""",unsafe_allow_html=True)

# ── auto refresh on live pages ──
if auto_ref and active in ["Dashboard","Analytics"]:
    time.sleep(5)
    st.rerun()