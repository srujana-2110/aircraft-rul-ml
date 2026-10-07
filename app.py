import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Aircraft Engine RUL Monitor",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------
# Load model and feature list
# --------------------------------------------------
model = joblib.load("aircraft_rul_random_forest.pkl")
features = joblib.load("rul_features.pkl")

# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f7fa;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")

    st.write(
        "This application predicts the Remaining Useful Life (RUL) "
        "of a simulated aircraft engine using Machine Learning."
    )

    st.subheader("Model")
    st.write("Random Forest Regressor")

    st.subheader("Dataset")
    st.write("NASA C-MAPSS FD001")

    st.subheader("Output")
    st.write("Estimated Remaining Useful Life in engine cycles")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">✈️ Aircraft Engine RUL Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predictive Maintenance & Engine Condition Monitoring'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
<b>How to use:</b> Enter the current engine readings below.
The remaining technical parameters are automatically filled using
median values from the training dataset.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# User inputs
# --------------------------------------------------
st.header("🔧 Engine Condition Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    cycle = st.number_input(
        "Engine Cycle",
        min_value=1,
        max_value=362,
        value=104,
        step=1
    )
    st.caption("Observed range: **1 – 362 cycles**")

with col2:
    sensor_11 = st.number_input(
        "Thermal Indicator (Sensor 11)",
        min_value=46.850,
        max_value=48.530,
        value=47.510,
        step=0.001,
        format="%.3f"
    )
    st.caption("Observed range: **46.850 – 48.530**")

with col3:
    sensor_9 = st.number_input(
        "Pressure/Performance Indicator (Sensor 9)",
        min_value=9021.730,
        max_value=9244.590,
        value=9060.660,
        step=0.001,
        format="%.3f"
    )
    st.caption("Observed range: **9021.730 – 9244.590**")

col4, col5, col6 = st.columns(3)

with col4:
    sensor_4 = st.number_input(
        "Engine Health Indicator (Sensor 4)",
        min_value=1382.250,
        max_value=1441.490,
        value=1408.040,
        step=0.001,
        format="%.3f"
    )
    st.caption("Observed range: **1382.250 – 1441.490**")

with col5:
    sensor_14 = st.number_input(
        "Performance Indicator (Sensor 14)",
        min_value=8099.940,
        max_value=8293.720,
        value=8140.540,
        step=0.001,
        format="%.3f"
    )
    st.caption("Observed range: **8099.940 – 8293.720**")

with col6:
    sensor_12 = st.number_input(
        "Condition Indicator (Sensor 12)",
        min_value=518.690,
        max_value=523.380,
        value=521.480,
        step=0.001,
        format="%.3f"
    )
    st.caption("Observed range: **518.690 – 523.380**")

st.caption(
    "⚠️ Sensor values are dataset-specific indicators and are not "
    "physical PSI, °C, bar, or other engineering units."
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button(
    "🔮 Predict Remaining Useful Life",
    use_container_width=True
):

    # --------------------------------------------------
    # Median values from the training dataset
    # --------------------------------------------------
    default_values = {
        "cycle": 104.0,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "sensor_2": 642.64,
        "sensor_3": 1590.1,
        "sensor_4": 1408.04,
        "sensor_6": 21.61,
        "sensor_7": 553.44,
        "sensor_8": 2388.09,
        "sensor_9": 9060.66,
        "sensor_11": 47.51,
        "sensor_12": 521.48,
        "sensor_13": 2388.09,
        "sensor_14": 8140.54,
        "sensor_15": 8.4389,
        "sensor_17": 393.0,
        "sensor_20": 38.83,
        "sensor_21": 23.2979
    }

    # --------------------------------------------------
    # Replace selected values with user inputs
    # --------------------------------------------------
    default_values["cycle"] = cycle
    default_values["sensor_11"] = sensor_11
    default_values["sensor_9"] = sensor_9
    default_values["sensor_4"] = sensor_4
    default_values["sensor_14"] = sensor_14
    default_values["sensor_12"] = sensor_12

    # --------------------------------------------------
    # Create model input in exact feature order
    # --------------------------------------------------
    input_data = pd.DataFrame(
        [[default_values[feature] for feature in features]],
        columns=features
    )

    # --------------------------------------------------
    # Predict RUL
    # --------------------------------------------------
    predicted_rul = float(model.predict(input_data)[0])

    # Prevent negative displayed RUL
    predicted_rul = max(0, predicted_rul)

    # --------------------------------------------------
    # Determine engine condition
    # --------------------------------------------------
    if predicted_rul > 50:
        condition = "Healthy"
        risk = "Low"
        recommendation = (
            "Continue routine monitoring and scheduled maintenance."
        )

    elif predicted_rul >= 20:
        condition = "Warning"
        risk = "Medium"
        recommendation = (
            "Increase monitoring frequency and schedule preventive "
            "maintenance."
        )

    else:
        condition = "Critical"
        risk = "High"
        recommendation = (
            "Immediate inspection recommended. Prioritize maintenance "
            "planning."
        )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------
    st.header("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Predicted RUL",
            f"{predicted_rul:.1f} cycles"
        )

    with result_col2:
        st.metric(
            "Engine Condition",
            condition
        )

    with result_col3:
        st.metric(
            "Risk Level",
            risk
        )

    # --------------------------------------------------
    # Maintenance recommendation
    # --------------------------------------------------
    st.subheader("🛠️ Maintenance Recommendation")

    if condition == "Healthy":
        st.success(recommendation)

    elif condition == "Warning":
        st.warning(recommendation)

    else:
        st.error(recommendation)

    st.info(
        "RUL is an estimated number of operating cycles remaining "
        "based on patterns learned from the NASA C-MAPSS FD001 dataset."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    '<div class="footer">'
    'NASA C-MAPSS FD001 • Random Forest Machine Learning • '
    'Predictive Maintenance Demo'
    '</div>',
    unsafe_allow_html=True
)
