import streamlit as st
import pandas as pd
import random
from math import radians, sin, cos, atan2, sqrt
from datetime import datetime

# ---------------- Folium Imports ----------------
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    folium = None
    def st_folium(map_obj, width=700, height=500):
        return {}

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Ola Bike Kenya",
    page_icon="https://cdn-icons-png.flaticon.com/512/2972/2972185.png",
    layout="wide"
)

# ---------------- Background ----------------
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.un.org/sites/un2.un.org/files/field/image/2024/06/ocean-2.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .leaflet-container {
        background: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Utility Functions ----------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def estimate_arrival_time(distance_km, speed_kmph=40):
    return (distance_km / speed_kmph) * 60 if speed_kmph > 0 else 0

# ---------------- County Data ----------------
@st.cache_data
def load_county_data():
    data = {
        "county": [
            "Mombasa","Kwale","Kilifi","Tana River","Lamu","Taita-Taveta","Garissa","Wajir",
            "Mandera","Marsabit","Isiolo","Meru","Tharaka-Nithi","Embu","Kitui","Machakos",
            "Makueni","Nyandarua","Nyeri","Kirinyaga","Murang'a","Kiambu","Turkana","West Pokot",
            "Samburu","Trans Nzoia","Uasin Gishu","Elgeyo-Marakwet","Nandi","Baringo","Laikipia",
            "Nakuru","Narok","Kajiado","Kericho","Bomet","Kakamega","Vihiga","Bungoma","Busia",
            "Siaya","Kisumu","Homa Bay","Migori","Kisii","Nyamira","Nairobi"
        ],
        "latitude":[
            -4.0435,-4.1710,-3.5107,-1.8936,-2.2717,-3.3166,-0.4569,1.7500,3.9373,2.3399,0.3524,0.0511,
            -0.2967,-0.5343,-1.3667,-1.5167,-1.8030,-0.2700,-0.4197,-0.6591,-0.7830,-1.1700,3.1190,1.5000,
            1.1056,1.0167,0.5143,0.7500,0.1833,0.4667,0.2000,-0.3031,-1.0800,-1.4500,-0.3670,
            -0.7833,0.2833,0.0500,0.5667,0.4600,0.0617,-0.0917,-0.5167,-1.0667,-0.6817,-0.5667,-1.2921
        ],
        "longitude":[
            39.6682,39.4521,39.8456,40.0974,40.9020,38.4849,39.6583,40.0670,41.8670,37.9980,37.5822,37.6456,
            37.7236,37.4500,38.0167,37.2667,37.6200,36.3500,36.9476,37.3827,37.0500,36.8356,35.6000,35.1000,
            36.7256,35.0167,35.2698,35.5833,35.0000,36.0833,36.5000,36.0800,35.8600,36.7833,35.2833,
            35.3500,34.7500,34.7333,34.5667,34.1167,34.2881,34.7679,34.4500,34.4667,34.7667,34.9333,36.8219
        ]
    }
    return pd.DataFrame(data)

county_df = load_county_data()

# ---------------- Session State ----------------
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

def save_feedback(name, county, feedback, rating):
    st.session_state.feedbacks.append({
        "Name": name,
        "County": county,
        "Feedback": feedback,
        "Rating": rating,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# ---------------- Main App ----------------
page = st.sidebar.radio("Go to:", ["Home", "Request Trip", "Map", "Feedback", "View Feedback"])

if page == "Home":
    st.title("🚴 Ola Bike Kenya")
    st.markdown("""
    Ola Bike Kenya is your **fast, affordable, and eco-friendly ride solution** connecting all **47 counties** of Kenya.  

    ### 💡 Why Choose Us?
    - ⚡ Quick Rides – Book in seconds, ride in minutes  
    - 🌍 Nationwide Reach – From Nairobi to Turkana, Mombasa to Kisii  
    - 💸 Affordable Fares – Transparent pricing with no hidden costs  
    - ♻️ Eco-Friendly – Reducing emissions, one ride at a time  
    - 🔒 Safe & Reliable – Trusted riders, real-time tracking & secure payments
    """)

elif page == "Request Trip":
    st.header("📌 Request a Ride")
    counties = county_df["county"].tolist()
    pickup = st.selectbox("Pickup County", counties)
    dropoff = st.selectbox("Dropoff County", counties)
    ride_time = st.time_input("Preferred Ride Time", value=datetime.now().time())
    payment_method = st.selectbox("Payment Method", ["Cash","Card","M-Pesa","Airtel Money","PayPal"])

    if pickup == dropoff:
        st.warning("⚠️ Pickup and dropoff counties must be different.")
    else:
        row_p = county_df[county_df["county"]==pickup].iloc[0]
        row_d = county_df[county_df["county"]==dropoff].iloc[0]
        lat1, lon1 = row_p.latitude, row_p.longitude
        lat2, lon2 = row_d.latitude, row_d.longitude

        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
        fare = 50 + (20*distance_km)
        arrival_time = int(random.uniform(1,5))
        trip_time = int(estimate_arrival_time(distance_km, 60))

        st.success(f"📏 Distance: **{distance_km:.2f} km**  💰 Fare: **Ksh {fare:.2f}**  ⏱️ Rider arriving in ~{arrival_time} mins  🛣️ Trip Duration: {trip_time} mins")
        
        if st.button("✅ Confirm Ride"):
            st.success(f"Ride confirmed from **{pickup}** → **{dropoff}** at {ride_time}. Enjoy your trip! 🚴💨")

elif page == "Map":
    st.header("🗺️ Interactive Map of Counties")
    if FOLIUM_AVAILABLE:
        m = folium.Map(location=[0.0236,37.9062], zoom_start=6, tiles="CartoDB positron")
        for _, row in county_df.iterrows():
            folium.Marker([row.latitude,row.longitude], popup=row.county).add_to(m)
        st_folium(m, width=900, height=600)
    else:
        st.info("Map feature is not available.")

elif page == "Feedback":
    st.header("📝 Give Your Feedback")
    name = st.text_input("Your Name")
    county = st.selectbox("County", county_df["county"].tolist())
    feedback = st.text_area("How was your trip?")
    rating = st.slider("Rate your experience ⭐", 1,5,5)
    if st.button("Submit Feedback"):
        if name.strip() and feedback.strip():
            save_feedback(name, county, feedback, rating)
            st.success("✅ Thank you for your feedback!")
        else:
            st.error("⚠️ Please fill all fields.")

elif page == "View Feedback":
    st.header("📊 Rider & Trip Feedback")
    if st.session_state.feedbacks:
        df_feedback = pd.DataFrame(st.session_state.feedbacks)
        st.dataframe(df_feedback)
    else:
        st.info("No feedback yet. Be the first to leave a review!")
