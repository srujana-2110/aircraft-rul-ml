import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aircraft Engine RUL Monitor",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("aircraft_rul_random_forest.pkl")
features = joblib.load("rul_features.pkl")


# ============================================================
# TRAINING MEDIAN VALUES
# ============================================================

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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("✈️ Aircraft RUL Monitor")

    st.markdown("---")

    st.subheader("📌 Project")

    st.write(
        "Aircraft Engine Remaining Useful Life "
        "(RUL) Prediction and Condition Monitoring "
        "using Machine Learning."
    )

    st.markdown("---")

    st.subheader("🤖 Model")

    st.write("Random Forest Regressor")
    st.write("18 model features")
    st.write("6 user inputs")

    st.markdown("---")

    st.subheader("📊 Dataset")

    st.write("NASA C-MAPSS FD001")
    st.write("100 training engines")
    st.write("1 operating condition")

    st.markdown("---")

    st.subheader("📈 Output")

    st.write("Remaining Useful Life")
    st.write("Engine Condition")
    st.write("Risk Level")
    st.write("Maintenance Recommendation")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("✈️ Aircraft Engine RUL Prediction")

st.subheader(
    "Predictive Maintenance & Engine Condition Monitoring"
)

st.write(
    "Enter the current engine parameters below to "
    "estimate the remaining useful life of the engine."
)


# ============================================================
# ENGINE PARAMETERS
# ============================================================

st.markdown("---")

st.subheader("🔧 Engine Parameters")


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# ENGINE CYCLE
# ------------------------------------------------------------

with col1:

    cycle = st.number_input(
        "Engine Cycle",
        min_value=1.0,
        max_value=362.0,
        value=104.0,
        step=1.0
    )

    st.caption(
        "Range: 1 – 362 | Typical: 104"
    )


# ------------------------------------------------------------
# SENSOR 11
# ------------------------------------------------------------

with col2:

    sensor_11 = st.number_input(
        "Thermal Indicator (Sensor 11)",
        min_value=46.850,
        max_value=48.530,
        value=47.510,
        step=0.001
    )

    st.caption(
        "Range: 46.850 – 48.530 | Typical: 47.510"
    )


# ------------------------------------------------------------
# SENSOR 9
# ------------------------------------------------------------

with col3:

    sensor_9 = st.number_input(
        "Pressure/Performance Indicator (Sensor 9)",
        min_value=9021.730,
        max_value=9244.590,
        value=9060.660,
        step=0.001
    )

    st.caption(
        "Range: 9021.730 – 9244.590 | Typical: 9060.660"
    )


col4, col5, col6 = st.columns(3)


# ------------------------------------------------------------
# SENSOR 4
# ------------------------------------------------------------

with col4:

    sensor_4 = st.number_input(
        "Engine Health Indicator (Sensor 4)",
        min_value=1382.250,
        max_value=1441.490,
        value=1408.040,
        step=0.001
    )

    st.caption(
        "Range: 1382.250 – 1441.490 | Typical: 1408.040"
    )


# ------------------------------------------------------------
# SENSOR 14
# ------------------------------------------------------------

with col5:

    sensor_14 = st.number_input(
        "Performance Indicator (Sensor 14)",
        min_value=8099.940,
        max_value=8293.720,
        value=8140.540,
        step=0.001
    )

    st.caption(
        "Range: 8099.940 – 8293.720 | Typical: 8140.540"
    )


# ------------------------------------------------------------
# SENSOR 12
# ------------------------------------------------------------

with col6:

    sensor_12 = st.number_input(
        "Condition Indicator (Sensor 12)",
        min_value=518.690,
        max_value=523.380,
        value=521.480,
        step=0.001
    )

    st.caption(
        "Range: 518.690 – 523.380 | Typical: 521.480"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Engine RUL",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    input_data = default_values.copy()

    input_data["cycle"] = cycle
    input_data["sensor_11"] = sensor_11
    input_data["sensor_9"] = sensor_9
    input_data["sensor_4"] = sensor_4
    input_data["sensor_14"] = sensor_14
    input_data["sensor_12"] = sensor_12

    input_df = pd.DataFrame([input_data])

    input_df = input_df[features]


    # --------------------------------------------------------
    # PREDICT RUL
    # --------------------------------------------------------

    predicted_rul = model.predict(input_df)[0]

    predicted_rul = max(0, predicted_rul)

    rul_cycles = round(predicted_rul)


    # ========================================================
    # ENGINE CONDITION
    # ========================================================

    if predicted_rul > 50:

        condition = "Healthy"
        risk = "Low"
        icon = "🟢"

        recommendation = (
            "Continue routine monitoring and follow "
            "the scheduled maintenance plan."
        )

        status_message = (
            "The engine is currently operating within "
            "a healthy RUL range."
        )

        progress_value = 100


    elif predicted_rul >= 20:

        condition = "Warning"
        risk = "Medium"
        icon = "🟡"

        recommendation = (
            "Increase monitoring frequency and schedule "
            "preventive maintenance."
        )

        status_message = (
            "The engine shows a moderate remaining life. "
            "Preventive maintenance should be planned."
        )

        progress_value = int(
            (predicted_rul / 50) * 100
        )


    else:

        condition = "Critical"
        risk = "High"
        icon = "🔴"

        recommendation = (
            "Immediate engine inspection is recommended. "
            "Prioritize maintenance planning."
        )

        status_message = (
            "The engine has limited remaining useful life. "
            "Immediate attention is recommended."
        )

        progress_value = int(
            (predicted_rul / 20) * 100
        )


    progress_value = max(
        0,
        min(progress_value, 100)
    )


    # ========================================================
    # ENGINE MONITORING DASHBOARD
    # ========================================================

    st.markdown("---")

    st.title("📊 Engine Health Dashboard")

    st.write(
        "Current engine condition based on the predicted "
        "Remaining Useful Life."
    )


    # ========================================================
    # MAIN KPI CARDS
    # ========================================================

    kpi1, kpi2, kpi3 = st.columns(3)


    with kpi1:

        st.metric(
            label="✈️ Remaining Useful Life",
            value=f"{rul_cycles} cycles"
        )


    with kpi2:

        st.metric(
            label="Engine Condition",
            value=f"{icon} {condition}"
        )


    with kpi3:

        st.metric(
            label="⚠️ Risk Level",
            value=risk
        )


    # ========================================================
    # ENGINE LIFE VISUAL
    # ========================================================

    st.markdown("---")

    st.subheader("🔋 Remaining Engine Life")

    st.progress(progress_value)

    st.caption(
        f"Estimated remaining useful life: "
        f"{rul_cycles} cycles"
    )


    # ========================================================
    # CONDITION STATUS
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"{icon} Engine Status: {condition}"
    )

    if condition == "Healthy":

        st.success(
            f"**Healthy Condition**\n\n"
            f"{status_message}"
        )

    elif condition == "Warning":

        st.warning(
            f"**Warning Condition**\n\n"
            f"{status_message}"
        )

    else:

        st.error(
            f"**Critical Condition**\n\n"
            f"{status_message}"
        )


    # ========================================================
    # MAINTENANCE RECOMMENDATION
    # ========================================================

    st.markdown("---")

    st.subheader("🛠️ Maintenance Recommendation")

    st.info(
        recommendation
    )


    # ========================================================
    # CURRENT PARAMETERS DASHBOARD
    # ========================================================

    st.markdown("---")

    st.subheader("📋 Current Engine Parameters")


    p1, p2, p3 = st.columns(3)


    with p1:

        st.metric(
            "Engine Cycle",
            f"{cycle:.0f}"
        )

        st.metric(
            "Thermal Indicator",
            f"{sensor_11:.3f}"
        )


    with p2:

        st.metric(
            "Pressure/Performance",
            f"{sensor_9:.3f}"
        )

        st.metric(
            "Engine Health",
            f"{sensor_4:.3f}"
        )


    with p3:

        st.metric(
            "Performance Indicator",
            f"{sensor_14:.3f}"
        )

        st.metric(
            "Condition Indicator",
            f"{sensor_12:.3f}"
        )


    # ========================================================
    # DECISION SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader("📝 Monitoring Summary")

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            f"**Predicted RUL:** {rul_cycles} cycles"
        )

        st.write(
            f"**Condition:** {icon} {condition}"
        )

        st.write(
            f"**Risk Level:** {risk}"
        )


    with summary_col2:

        st.write(
            "**Recommended Action:**"
        )

        st.write(
            recommendation
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dataset: NASA C-MAPSS FD001 | "
    "Model: Random Forest Regressor"
)

st.caption(
    "Aircraft Engine Remaining Useful Life Prediction "
    "and Condition Monitoring System"
)
