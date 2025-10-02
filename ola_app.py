# ola_app.py
import streamlit as st
import pandas as pd
import pickle, os, json
from math import radians, sin, cos, atan2, sqrt
from datetime import datetime

# ---------------- Folium Imports ----------------
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False
    folium = None

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Ola Bike Kenya",
    page_icon="https://cdn-icons-png.flaticon.com/512/2972/2972185.png",
    layout="wide"
)

# ---------------- Background Styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.un.org/sites/un2.un.org/files/field/image/2024/06/ocean-2.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #0b2e4f;
    }
    .leaflet-container {
        background: white !important;
        border-radius: 12px;
    }
    .main-container .block-container {
        background: rgba(255,255,255,0.92);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Persistence Setup ----------------
DATA_FILE = "ola_data.pkl"

def save_data():
    """Save trips, riders, and feedbacks to pickle file."""
    with open(DATA_FILE, "wb") as f:
        pickle.dump({
            "trips": st.session_state.trips,
            "rider_applications": st.session_state.rider_applications,
            "feedbacks": st.session_state.feedbacks
        }, f)

def load_data():
    """Load trips, riders, and feedbacks from pickle file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
            st.session_state.trips = data.get("trips", [])
            st.session_state.rider_applications = data.get("rider_applications", [])
            st.session_state.feedbacks = data.get("feedbacks", [])

# ---------------- Data storage in session ----------------
if "trips" not in st.session_state:
    st.session_state.trips = []
if "rider_applications" not in st.session_state:
    st.session_state.rider_applications = []
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

# Load saved data at startup
load_data()

# ---------------- Utility functions ----------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def estimate_arrival_time(distance_km, speed_kmph=40):
    if speed_kmph <= 0:
        return 0
    return (distance_km / speed_kmph) * 60

def calculate_fare(distance_km, base_fare=50, per_km=20):
    return round(base_fare + (distance_km * per_km), 2)

# ---------------- Locations dictionary ----------------
locations = {
    "Mombasa": {
        "Mombasa CBD": [-4.0435, 39.6682],
        "Nyali": [-4.0433, 39.68],
        "Likoni": [-4.1, 39.6667],
        "Changamwe": [-4.0167, 39.6167],
        "Kisauni": [-4.017, 39.7],
        "Bamburi": [-4.016, 39.72],
        "Shanzu": [-4.02, 39.73],
        "Mtwapa": [-3.95, 39.75],
        "Jomvu": [-4.05, 39.6],
        "Port Reitz": [-4.016, 39.58]
    },
    "Kwale": {
        "Kwale Town 1": [-0.54, 37.03],
        "Kwale Town 2": [-0.53, 37.04],
        "Kwale Town 3": [-0.52, 37.05],
        "Kwale Town 4": [-0.51, 37.06],
        "Kwale Town 5": [-0.5, 37.07],
        "Kwale Town 6": [-0.49, 37.08],
        "Kwale Town 7": [-0.48, 37.09],
        "Kwale Town 8": [-0.47, 37.1],
        "Kwale Town 9": [-0.46, 37.11],
        "Kwale Town 10": [-0.45, 37.12]
    },
    "Nairobi": {
        "Nairobi CBD": [-1.2833, 36.8167],
        "Westlands": [-1.2654, 36.811],
        "Karen": [-1.3398, 36.7176],
        "Lang'ata": [-1.3667, 36.75],
        "Embakasi": [-1.3333, 36.9],
        "Dagoretti": [-1.3167, 36.7333],
        "Kasarani": [-1.22, 36.9],
        "Ruaraka": [-1.25, 36.8667],
        "Gikambura": [-1.2333, 36.75],
        "South B": [-1.3167, 36.85]
    }
}

sorted_counties = sorted(list(locations.keys()))

# ---------------- Rider Application ----------------
def rider_application_page():
    st.title("🚴 Apply as a Rider")

    with st.form("rider_form", clear_on_submit=False):
        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        county = st.selectbox("County", sorted_counties, key="rider_county_select")
        town = st.selectbox("Town / City", list(locations[county].keys()), key="rider_town_select")
        license_no = st.text_input("Driving License Number")
        years_exp = st.number_input("Years of Riding Experience", min_value=0, max_value=60, value=1, step=1)

        submitted = st.form_submit_button("Apply as Rider")
        if submitted:
            if not name.strip() or not phone.strip() or not license_no.strip():
                st.error("⚠️ Please fill all required fields.")
            else:
                st.session_state.rider_applications.append({
                    "Name": name.strip(),
                    "Phone": phone.strip(),
                    "County": county,
                    "Town": town,
                    "License Number": license_no.strip(),
                    "Years Experience": int(years_exp),
                    "Applied On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_data()
                st.success(f"✅ Application submitted. Thank you, {name.split()[0]}!")

# ---------------- Request Trip Page ----------------
def request_trip_page():
    st.title("📌 Request a Ride")

    col1, col2 = st.columns(2)
    with col1:
        pickup_county = st.selectbox("Pickup County", sorted_counties, key="pickup_county")
        pickup_town = st.selectbox("Pickup Town/City", list(locations[pickup_county].keys()), key="pickup_town")
    with col2:
        drop_county = st.selectbox("Dropoff County", sorted_counties, key="drop_county")
        drop_town = st.selectbox("Dropoff Town/City", list(locations[drop_county].keys()), key="drop_town")

    pickup_lat, pickup_lon = locations[pickup_county][pickup_town]
    drop_lat, drop_lon = locations[drop_county][drop_town]

    if st.button("Estimate Ride"):
        if pickup_county == drop_county and pickup_town == drop_town:
            st.error("⚠️ Pickup and dropoff cannot be the same place.")
        else:
            distance_km = haversine_distance(pickup_lat, pickup_lon, drop_lat, drop_lon)
            fare = calculate_fare(distance_km)
            eta_min = int(estimate_arrival_time(distance_km, speed_kmph=60))

            st.success(f"📏 Distance: {distance_km:.2f} km | 💰 Fare: Ksh {fare:.2f} | ⏱ ETA: {eta_min} mins")

            if st.button("✅ Confirm Ride"):
                st.session_state.trips.append({
                    "Pickup": f"{pickup_town}, {pickup_county}",
                    "Dropoff": f"{drop_town}, {drop_county}",
                    "Distance (km)": round(distance_km, 2),
                    "Fare": fare,
                    "ETA (min)": eta_min,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_data()
                st.success("Ride confirmed! 🚲")

            if FOLIUM_AVAILABLE:
                mid_lat = (pickup_lat + drop_lat) / 2
                mid_lon = (pickup_lon + drop_lon) / 2
                m = folium.Map(location=[mid_lat, mid_lon], zoom_start=7, tiles="CartoDB positron")
                folium.Marker([pickup_lat, pickup_lon], popup=f"Pickup: {pickup_town}", icon=folium.Icon(color="green")).add_to(m)
                folium.Marker([drop_lat, drop_lon], popup=f"Dropoff: {drop_town}", icon=folium.Icon(color="red")).add_to(m)
                st_folium(m, width=900, height=500)

# ---------------- Feedback Page ----------------
def feedback_page():
    st.title("📝 Feedback")

    name = st.text_input("Your Name", key="feedback_name")
    rating = st.slider("Rating", 1, 5, 5, key="feedback_rating")
    feedback = st.text_area("Feedback", key="feedback_text")

    if st.button("Submit Feedback"):
        if not name.strip() or not feedback.strip():
            st.error("⚠️ Please provide your name and feedback.")
        else:
            st.session_state.feedbacks.append({
                "Name": name.strip(),
                "Rating": rating,
                "Feedback": feedback.strip(),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_data()
            st.success("✅ Thank you for your feedback!")

# ---------------- Admin Dashboard ----------------
def admin_dashboard():
    st.title("📊 Admin Dashboard")
    section = st.radio("View:", ["Trips", "Riders", "Feedbacks"], horizontal=True)

    if section == "Trips":
        if st.session_state.trips:
            df = pd.DataFrame(st.session_state.trips)
            st.dataframe(df)
            total = df["Fare"].sum()
            st.success(f"💰 Total Revenue: Ksh {total:.2f}")
        else:
            st.info("No trips yet.")

    elif section == "Riders":
        if st.session_state.rider_applications:
            df = pd.DataFrame(st.session_state.rider_applications)
            st.dataframe(df)
        else:
            st.info("No riders yet.")

    else:
        if st.session_state.feedbacks:
            df = pd.DataFrame(st.session_state.feedbacks)
            st.dataframe(df)
        else:
            st.info("No feedbacks yet.")

# ---------------- Main navigation ----------------
st.sidebar.title("🚴 Ola Bike Kenya")
page = st.sidebar.radio("Go to:", ["Home", "Request Trip", "Apply as Rider", "Feedback", "Admin Dashboard"])

if page == "Home":
    st.title("🚴 Ola Bike Kenya")
    st.markdown("**Ola Bike Kenya** — Swift, affordable, and eco-friendly rides across 47 counties.")
elif page == "Request Trip":
    request_trip_page()
elif page == "Apply as Rider":
    rider_application_page()
elif page == "Feedback":
    feedback_page()
elif page == "Admin Dashboard":
    admin_dashboard()
