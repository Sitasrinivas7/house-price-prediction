import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")

FEATURES = [
    "location",
    "Carpet Area (sqft)",
    "Status",
    "Floor",
    "Transaction",
    "Furnishing",
    "Bathroom",
    "Balcony",
    "Car Parking",
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "house_price_model.pkl was not found. Put it in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #153e75, #2878a8);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin-bottom: .35rem;}
    .hero p {color: #eaf4ff; margin-bottom: 0;}
    .result-card {
        padding: 1.2rem; border-radius: 16px;
        background: rgba(49, 130, 206, 0.10);
        border: 1px solid rgba(49, 130, 206, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🏠 House Price Prediction</h1>
      <p>Enter property details to get an estimated price from the trained Random Forest model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This is an educational machine-learning project. Predictions are estimates, "
    "not professional valuations or guaranteed market prices."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.markdown(
        """
        **Setup check**
        1. Make sure `house_price_model.pkl` is in the same folder as `app.py`.
        2. Install the packages listed in `requirements.txt`.
        3. If the error mentions a scikit-learn version mismatch, use the same
           scikit-learn version that was used to save the model.
        """
    )
    st.stop()

st.subheader("Enter property details")
with st.form("property_form"):
    left, middle, right = st.columns(3)

    with left:
        location = st.text_input("Location", value="thane", help="Enter the location as represented in the dataset.")
        carpet_area = st.number_input(
            "Carpet area (sqft)",
            min_value=1.0,
            max_value=100000.0,
            value=700.0,
            step=25.0,
        )
        status = st.selectbox(
            "Property status",
            ["Ready to Move", "Under Construction", "Unknown"],
        )

    with middle:
        floor = st.text_input("Floor", value="5 out of 20", help='Example: "5 out of 20"')
        transaction = st.selectbox("Transaction", ["Resale", "New Property", "Unknown"])
        furnishing = st.selectbox(
            "Furnishing",
            ["Unfurnished", "Semi-Furnished", "Furnished", "Unknown"],
        )

    with right:
        bathroom = st.selectbox("Bathrooms", ["1", "2", "3", "4", "5", "6", "Unknown"], index=1)
        balcony = st.selectbox("Balconies", ["0", "1", "2", "3", "4", "Unknown"], index=1)
        car_parking = st.selectbox(
            "Car parking",
            ["Unknown", "No Parking", "1 Open", "1 Covered", "2 Open", "2 Covered"],
            index=3,
        )

    submitted = st.form_submit_button("🔎 Predict house price", use_container_width=True)

if submitted:
    if not location.strip():
        st.error("Please enter a location.")
    elif not floor.strip():
        st.error("Please enter the floor details.")
    else:
        input_data = pd.DataFrame(
            [{
                "location": location.strip().lower(),
                "Carpet Area (sqft)": float(carpet_area),
                "Status": status,
                "Floor": floor.strip(),
                "Transaction": transaction,
                "Furnishing": furnishing,
                "Bathroom": bathroom,
                "Balcony": balcony,
                "Car Parking": car_parking,
            }],
            columns=FEATURES,
        )

        try:
            prediction = float(model.predict(input_data)[0])
            if prediction < 0:
                st.warning("The model returned a negative estimate. Treat this prediction as unreliable.")
            else:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.caption("Estimated property price")
                st.markdown(f"## ₹{prediction:,.0f}")
                st.markdown(f"**Approximately ₹{prediction / 100000:.2f} lakhs**")
                st.markdown(f"**Approximately ₹{prediction / 10000000:.2f} crore**")
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption(
                    "The result depends on the training data and the input categories. "
                    "Verify against current local listings before making decisions."
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with st.expander("About this project"):
    st.write(
        "The notebook trains a Linear Regression baseline and a Random Forest Regressor. "
        "The saved pipeline includes preprocessing and the Random Forest model."
    )
    st.write(
        "Model inputs: location, carpet area, status, floor, transaction, furnishing, "
        "bathrooms, balconies, and car parking."
    )
