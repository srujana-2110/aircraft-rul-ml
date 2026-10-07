
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Aircraft Engine RUL Monitor",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("aircraft_rul_random_forest.pkl")
features = joblib.load("rul_features.pkl")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
    background-color: #f8f9fa;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("✈️ RUL Monitor")

    st.markdown("""
    ### About the Project

    This application predicts the **Remaining Useful Life (RUL)**
    of an aircraft engine using a Random Forest Machine Learning model.

    The system also provides:

    - Engine condition
    - Risk level
    - Maintenance recommendation
    """)

    st.divider()

    st.markdown("### Model")

    st.write("Random Forest Regressor")

    st.markdown("### Dataset")

    st.write("NASA C-MAPSS FD001")

    st.divider()

    st.caption(
        "Educational project using simulated aircraft engine data."
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">✈️ Aircraft Engine RUL Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predictive Maintenance & Engine Condition Monitoring</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter the latest engine operating and sensor measurements "
    "to estimate the engine's Remaining Useful Life."
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">⚙️ Engine Parameters</div>',
    unsafe_allow_html=True
)

input_values = {}

# Create three columns
col1, col2, col3 = st.columns(3)

for i, feature in enumerate(features):

    # Reasonable starting value
    if feature == "cycle":
        default_value = 100.0
    else:
        default_value = 0.0

    if i % 3 == 0:

        with col1:
            input_values[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.4f"
            )

    elif i % 3 == 1:

        with col2:
            input_values[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.4f"
            )

    else:

        with col3:
            input_values[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.4f"
            )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predict_button = st.button(
    "🔍 Predict Engine RUL",
    type="primary",
    use_container_width=True
)

if predict_button:

    input_df = pd.DataFrame(
        [input_values],
        columns=features
    )

    predicted_rul = model.predict(input_df)[0]

    # Prevent negative RUL display
    predicted_rul = max(0, predicted_rul)

    # --------------------------------------------------
    # CONDITION CLASSIFICATION
    # --------------------------------------------------

    if predicted_rul > 50:

        condition = "Healthy"
        risk = "Low"
        recommendation = (
            "Continue routine monitoring and scheduled maintenance."
        )
        emoji = "🟢"

    elif predicted_rul >= 20:

        condition = "Warning"
        risk = "Medium"
        recommendation = (
            "Increase monitoring frequency and schedule "
            "preventive maintenance."
        )
        emoji = "🟡"

    else:

        condition = "Critical"
        risk = "High"
        recommendation = (
            "Immediate inspection recommended. "
            "Prioritize maintenance planning."
        )
        emoji = "🔴"

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Prediction Results</div>',
        unsafe_allow_html=True
    )

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Remaining Useful Life",
            f"{predicted_rul:.1f} cycles"
        )

    with result2:

        st.metric(
            "Engine Condition",
            f"{emoji} {condition}"
        )

    with result3:

        st.metric(
            "Risk Level",
            risk
        )

    st.divider()

    # --------------------------------------------------
    # MAINTENANCE RECOMMENDATION
    # --------------------------------------------------

    st.subheader("🔧 Maintenance Recommendation")

    if condition == "Healthy":

        st.success(recommendation)

    elif condition == "Warning":

        st.warning(recommendation)

    else:

        st.error(recommendation)

    # --------------------------------------------------
    # RUL INTERPRETATION
    # --------------------------------------------------

    st.subheader("📌 RUL Interpretation")

    if predicted_rul > 50:

        st.write(
            "The predicted RUL is above the project's healthy threshold. "
            "The engine can continue under routine monitoring."
        )

    elif predicted_rul >= 20:

        st.write(
            "The predicted RUL indicates a warning condition. "
            "Closer monitoring and preventive maintenance are recommended."
        )

    else:

        st.write(
            "The predicted RUL indicates a critical condition. "
            "Inspection and maintenance should be prioritized."
        )

    st.caption(
        "Note: Health thresholds are project-defined decision rules "
        "and are not official NASA condition labels."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Aircraft Engine RUL Prediction | "
    "Random Forest | NASA C-MAPSS FD001 | Streamlit"
)
