

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import time

# Load trained model path
model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "flight_delay_rf_model_updated.pkl"
)


# Page settings
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)


# Modern Dashboard Styling
st.markdown("""
<style>
    .stApp {
        background-color: #F5F8FF;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1250px;
    }

    h1, h2, h3 {
        color: #173B68;
    }

    div.stButton > button[kind="primary"] {
        background-color: #3478F6;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #205BC2;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E3EAF5;
    }
</style>
""", unsafe_allow_html=True)


# Dashboard Header
st.title("✈️ Flight Delay Dashboard")

st.write(
    "Predict flight delays using Machine Learning "
    "and explore your flight's estimated delay risk."
)

st.caption(
    "Model trained on 2024 US flight data. "
    "Predictions for future dates are estimates and "
    "may be less reliable due to changing flight patterns."
)


st.divider()

# Flight Information
st.subheader("Flight Information")

st.info(
    "Enter your flight details to generate a delay prediction."
)


@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Get actual airline and airport codes from trained model
encoder = model.named_steps["preprocessing"].named_transformers_[
    "categorical"
].named_steps["encoder"]

airline_codes = sorted(encoder.categories_[0].tolist())
origin_codes = sorted(encoder.categories_[1].tolist())
destination_codes = sorted(encoder.categories_[2].tolist())

print("Airlines:", len(airline_codes))
print("Departure Airports:", len(origin_codes))
print("Destination Airports:", len(destination_codes))


# Flight Information - Modern Layout

# Row 1: Flight Date and Departure Airport
col1, col2 = st.columns(2, gap="large")

with col1:
    flight_date = st.date_input("Flight Date")

    month = flight_date.month
    day_of_month = flight_date.day
    day_of_week = (flight_date.weekday() + 1) % 7 + 1

with col2:
    origin = st.selectbox(
        "Departure Airport",
        options=origin_codes,
        index=1
    )

# Row 2: Airline and Destination Airport
col3, col4 = st.columns(2, gap="large")

with col3:
    airline = st.selectbox(
        "Airline",
        options=airline_codes,
        index=airline_codes.index("AA") if "AA" in airline_codes else 0
    )

with col4:
    destination = st.selectbox(
        "Destination Airport",
        options=destination_codes,
        index=2
    )

# Row 3: Departure Time and Flight Distance
col5, col6 = st.columns(2, gap="large")

with col5:
    flight_time = st.time_input(
        "Scheduled Departure Time",
        value=time(12, 0)
    )

    departure_time = (
        flight_time.hour * 100 + flight_time.minute
    )

with col6:
    distance = st.number_input(
        "Flight Distance (miles)",
        min_value=1.0,
        value=500.0
    )


st.caption(
    "Note: Select a valid flight route. "
    "Not all airport combinations represent existing direct flights."
)


# Prediction Button
if st.button("✈️ Predict Flight Delay", type="primary"):

    # Validate airports
    if origin == destination:
        st.error("Departure and destination airports cannot be the same.")
        st.stop()

    # Prepare flight data
    flight_data = pd.DataFrame([{
        "month": month,
        "day_of_month": day_of_month,
        "day_of_week": day_of_week,
        "op_unique_carrier": airline.strip().upper(),
        "origin": origin.strip().upper(),
        "dest": destination.strip().upper(),
        "crs_dep_time": departure_time,
        "distance": distance
    }])

    # Make prediction
    probability = float(model.predict_proba(flight_data)[0][1])
    prediction = int(probability >= 0.50)

    # Determine flight status
    if prediction == 1:
        flight_status = "Delayed"
        status_color = "#D9534F"
    else:
        flight_status = "On Time"
        status_color = "#198754"

    # Determine risk level
    if probability >= 0.70:
        risk_level = "High"
        risk_color = "#D9534F"
    elif probability >= 0.40:
        risk_level = "Moderate"
        risk_color = "#E6A23C"
    else:
        risk_level = "Low"
        risk_color = "#198754"

    st.divider()
    st.subheader("Prediction Overview")

    # Three result cards
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            f"""
            <div style="background:white;padding:24px;
                        border-radius:16px;border:1px solid #E3EAF5;
                        min-height:125px;">
                <p style="color:#64748B;margin:0;">Flight Status</p>
                <h2 style="color:{status_color};margin-top:12px;">
                    {flight_status}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:white;padding:24px;
                        border-radius:16px;border:1px solid #E3EAF5;
                        min-height:125px;">
                <p style="color:#64748B;margin:0;">Delay Probability</p>
                <h2 style="color:#173B68;margin-top:12px;">
                    {probability * 100:.1f}%
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="background:white;padding:24px;
                        border-radius:16px;border:1px solid #E3EAF5;
                        min-height:125px;">
                <p style="color:#64748B;margin:0;">Risk Level</p>
                <h2 style="color:{risk_color};margin-top:12px;">
                    {risk_level}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.write("**Estimated Delay Probability**")
    st.progress(probability)

    st.caption(
        "A delay means departure is 15 minutes or more later "
        "than scheduled. This is a model estimate, "
        "not a guaranteed flight status."
    )

# Model Performance Section
st.divider()
st.subheader("📊 Model Performance")

st.caption(
    "Random Forest evaluated on 106,105 flights "
    "from October to December 2024."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", "72.56%")
m2.metric("Precision", "27.04%")
m3.metric("Recall", "37.89%")
m4.metric("F1-Score", "31.56%")


# Monthly Flight Delay Analysis
st.divider()
st.subheader("📈 Monthly Flight Delay Analysis")

# Dataset Summary Cards
s1, s2, s3 = st.columns(3)

s1.metric("Total Flights Analyzed", "6,982,766")

s2.metric(
    "Highest Delay Month",
    "July",
    help="Delay rate: 29.18%"
)

s3.metric(
    "Lowest Delay Month",
    "October",
    help="Delay rate: 13.73%"
)

st.caption(
    "Highest monthly delay rate: July (29.18%) | "
    "Lowest monthly delay rate: October (13.73%)"
)


monthly_data = pd.DataFrame({
    "Month": [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ],
    "Delay Rate (%)": [
        22.93, 15.41, 20.49, 19.04,
        25.65, 25.31, 29.18, 23.27,
        15.34, 13.73, 14.57, 21.85
    ]
})


# Display months in chronological order
st.bar_chart(
    monthly_data,
    x="Month",
    y="Delay Rate (%)",
    color="#3478F6",
    sort=False
)

