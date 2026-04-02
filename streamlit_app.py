import streamlit as st
import pandas as pd
import pickle
import base64
from pathlib import Path
from datetime import datetime, date, time

st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------
# Helpers
# -----------------------------
def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()


def compute_duration(dep_dt, arr_dt):
    diff = int((arr_dt - dep_dt).total_seconds() // 60)
    if diff < 0:
        diff += 1440
    hours = diff // 60
    mins = diff % 60
    return hours, mins


def build_features(dep_dt, arr_dt, stops, airline, source, destination):
    journey_day = dep_dt.day
    journey_month = dep_dt.month
    dep_hour = dep_dt.hour
    dep_min = dep_dt.minute
    arr_hour = arr_dt.hour
    arr_min = arr_dt.minute
    duration_h, duration_m = compute_duration(dep_dt, arr_dt)

    feature_dict = {
        "Total_Stops": stops,
        "journey_day": journey_day,
        "journey_month": journey_month,
        "dep_hour": dep_hour,
        "dep_min": dep_min,
        "arrival_hour": arr_hour,
        "arrival_min": arr_min,
        "Duration_hours": duration_h,
        "Duration_mins": duration_m,

        # Airline dummies
        "Airline_Air India": 1 if airline == "Air India" else 0,
        "Airline_GoAir": 1 if airline == "GoAir" else 0,
        "Airline_IndiGo": 1 if airline == "IndiGo" else 0,
        "Airline_Jet Airways": 1 if airline == "Jet Airways" else 0,
        "Airline_Multiple carriers": 1 if airline == "Multiple carriers" else 0,
        "Airline_Other": 1 if airline in [
            "Air Asia",
            "Jet Airways Business",
            "Multiple carriers Premium economy",
            "Trujet",
            "Vistara Premium economy",
        ] else 0,
        "Airline_SpiceJet": 1 if airline == "SpiceJet" else 0,
        "Airline_Vistara": 1 if airline == "Vistara" else 0,

        # Source dummies
        # Baseline is Banglore because drop_first=True and B comes first alphabetically
        "Source_Chennai": 1 if source == "Chennai" else 0,
        "Source_Delhi": 1 if source == "Delhi" else 0,
        "Source_Kolkata": 1 if source == "Kolkata" else 0,
        "Source_Mumbai": 1 if source == "Mumbai" else 0,

        # Destination dummies
        # Baseline is Banglore because drop_first=True and B comes first alphabetically
        "Destination_Cochin": 1 if destination == "Cochin" else 0,
        "Destination_Delhi": 1 if destination == "Delhi" else 0,
        "Destination_Hyderabad": 1 if destination == "Hyderabad" else 0,
        "Destination_Kolkata": 1 if destination == "Kolkata" else 0,
        "Destination_New Delhi": 1 if destination == "New Delhi" else 0,
    }

    X = pd.DataFrame([feature_dict])

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    return X

def load_model():
    candidates = [
        BASE_DIR / "c2_flight_rf.pkl",
        BASE_DIR / "c1_flight_rf.pkl"
    ]

    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


# -----------------------------
# Assets
# -----------------------------
banner_path = BASE_DIR / "plane_banner.jpg"
banner_base64 = get_base64_image(banner_path) if banner_path.exists() else ""


# -----------------------------
# Page Styling
# -----------------------------
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: "Segoe UI", sans-serif;
}}

.stApp {{
    background: linear-gradient(135deg, #7dd3fc 0%, #3b82f6 45%, #1d4ed8 75%, #1e3a8a 100%);
}}

.block-container {{
    margin-top: 24px;
    max-width: 1200px;
    padding-top: 1rem !important;
    padding-bottom: 2rem;
}}
          

.main-shell {{
    display: flex;
    justify-content: center;
    margin-top: 20px;   /* small top spacing */
}}

.form-shell {{
    width: 100%;
    max-width: 1120px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 28px;
    padding: 22px 22px 18px 22px;
    box-shadow: 0 22px 60px rgba(15, 23, 42, 0.28);
}}

.banner-wrap {{
    width: 100%;
    height: 180px;
    border-radius: 22px;
    overflow: hidden;
    margin-bottom: 24px;
    margin-top: 24px;
    box-shadow: 0 14px 35px rgba(0, 0, 0, 0.18);
}}

.banner-wrap img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}}

.inner-card {{
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 10px 30px rgba(15,23,42,0.14);
    height: 100%;
}}

.metric-card {{
    background: rgba(255,255,255,0.82);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    margin: 12px 0;
    color: #1f2937;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 8px 18px rgba(15,23,42,0.10);
}}

.metric-value {{
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
}}

.metric-label {{
    font-size: 15px;
    color: #334155;
}}

.section-title {{
    color: white;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 16px;
}}

.airline-pill {{
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(10px);
    color: white;
    border-radius: 16px;
    padding: 14px 16px;
    font-weight: 600;
}}

.result-card {{
    background: rgba(255,255,255,0.85);
    border-radius: 22px;
    padding: 18px 24px;
    text-align: center;
    color: #1e293b;
    box-shadow: 0 10px 28px rgba(15,23,42,0.12);
    max-width: 320px;
    margin: 0 auto;
}}

.result-price {{
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 4px;
}}

.footer-note {{
    text-align: center;
    color: rgba(255,255,255,0.92);
    margin-top: 14px;
    font-size: 14px;
}}

label, .stDateInput label, .stTimeInput label, .stSelectbox label, .stSlider label {{
    color: white !important;
    font-weight: 600 !important;
}}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stDateInput input,
.stTimeInput input {{
    background: rgba(255,255,255,0.88) !important;
    color: #0f172a !important;
    border-radius: 12px !important;
}}

.stSlider p {{
    color: white !important;
    font-weight: 600;
}}

.stButton > button {{
    width: 100%;
    height: 54px;
    border-radius: 14px;
    border: none;
    background: white;
    color: #1e3a8a;
    font-size: 18px;
    font-weight: 800;
    box-shadow: 0 8px 18px rgba(15,23,42,0.12);
}}

.stButton > button:hover {{
    background: #eff6ff;
    color: #1d4ed8;
}}

[data-testid="stMarkdownContainer"] p {{
    color: inherit;
}}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load model
# -----------------------------
model = load_model()
if model is None:
    st.error("Model could not be loaded.")
    st.stop()


# -----------------------------
# Constants
# -----------------------------
AIRLINES = [
    "Jet Airways",
    "IndiGo",
    "Air India",
    "Multiple carriers",
    "SpiceJet",
    "Vistara",
    "GoAir",
    "Air Asia",
    "Jet Airways Business",
    "Multiple carriers Premium economy",
    "Trujet",
    "Vistara Premium economy",
]

SOURCES = ["Delhi", "Kolkata", "Banglore", "Mumbai", "Chennai"]
DESTINATIONS = ["Cochin", "Banglore", "Delhi", "New Delhi", "Hyderabad", "Kolkata"]


# -----------------------------
# Outer shell
# -----------------------------
# st.markdown('<div class="main-shell"><div class="form-shell">', unsafe_allow_html=True)

if banner_base64:
    st.markdown(f"""
    <div class="banner-wrap">
        <img src="data:image/jpeg;base64,{banner_base64}">
    </div>
    """, unsafe_allow_html=True)

left, right = st.columns([1.25, 0.9], gap="large")

with left:
    # st.markdown('<div class="inner-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Trip details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        dep_date = st.date_input("Departure date", date.today())
        dep_time = st.time_input("Departure time", time(9, 0))
        source = st.selectbox("Source", SOURCES)
        stops_label = st.select_slider(
            "Total stops",
            options=["Non-stop", "1 stop", "2 stops", "3 stops", "4 stops"],
            value="Non-stop"
        )

    with c2:
        arr_date = st.date_input("Arrival date", date.today())
        arr_time = st.time_input("Arrival time", time(12, 0))
        destination = st.selectbox("Destination", DESTINATIONS)
        airline = st.selectbox("Preferred airline", AIRLINES)

    st.markdown('</div>', unsafe_allow_html=True)

dep_dt = datetime.combine(dep_date, dep_time)
arr_dt = datetime.combine(arr_date, arr_time)

stop_map = {
    "Non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
}
stops = stop_map[stops_label]
dur_h, dur_m = compute_duration(dep_dt, arr_dt)

with right:
    # st.markdown('<div class="inner-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Trip summary</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{source}</div>
            <div class="metric-label">Source</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{destination}</div>
            <div class="metric-label">Destination</div>
        </div>
        """, unsafe_allow_html=True)

    m3, m4 = st.columns(2)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stops}</div>
            <div class="metric-label">Total Stops</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{dur_h}h {dur_m}m</div>
            <div class="metric-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:20px; margin-bottom:10px;">Selected airline</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="airline-pill">{airline}</div>', unsafe_allow_html=True)

    if source == destination:
        st.warning("Source and destination are the same. Please check your selection.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

if st.button("Predict Fare"):
    try:
        X = build_features(dep_dt, arr_dt, stops, airline, source, destination)
        # st.write(list(model.feature_names_in_))
        pred = model.predict(X)[0]
        price = round(float(pred), 2)

        st.markdown("""
        <div style="margin-top:8px; margin-bottom:8px;">
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-price">₹ {price:,.2f}</div>
            <div>Estimated Ticket Price</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# st.markdown('<div class="footer-note">Built with Streamlit</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)