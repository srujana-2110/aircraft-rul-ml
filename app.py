import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
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
# SENSOR NORMAL RANGES
# ============================================================

sensor_ranges = {

    "sensor_4": {
        "name": "LPT Outlet Temperature",
        "min": 1382.250,
        "max": 1441.490,
        "median": 1408.040
    },

    "sensor_9": {
        "name": "Physical Core Speed",
        "min": 9021.730,
        "max": 9244.590,
        "median": 9060.660
    },

    "sensor_11": {
        "name": "HPC Outlet Static Pressure",
        "min": 46.850,
        "max": 48.530,
        "median": 47.510
    },

    "sensor_12": {
        "name": "Fuel Flow / Pressure Ratio",
        "min": 518.690,
        "max": 523.380,
        "median": 521.480
    },

    "sensor_14": {
        "name": "Corrected Core Speed",
        "min": 8099.940,
        "max": 8293.720,
        "median": 8140.540
    }
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("✈️ Aircraft RUL Monitor")

    st.markdown("---")

    st.subheader("Project")

    st.write(
        "Aircraft Engine Remaining Useful Life "
        "(RUL) Prediction and Condition Monitoring "
        "using Machine Learning."
    )

    st.markdown("---")

    st.subheader("Machine Learning Model")

    st.write("Random Forest Regressor")
    st.write("18 model features")
    st.write("50 decision trees")

    st.markdown("---")

    st.subheader("Dataset")

    st.write("NASA C-MAPSS FD001")
    st.write("100 training engines")
    st.write("1 operating condition")

    st.markdown("---")

    st.subheader("System Capabilities")

    st.write("✓ RUL Prediction")
    st.write("✓ Risk Assessment")
    st.write("✓ Sensor Monitoring")
    st.write("✓ Anomaly Detection")
    st.write("✓ Maintenance Recommendation")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("✈️ Aircraft Engine Predictive Maintenance")

st.subheader(
    "Remaining Useful Life Prediction & Condition Monitoring"
)

st.write(
    "Enter the current engine operating parameters to "
    "estimate remaining useful life, assess engine risk, "
    "and generate a maintenance recommendation."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("---")

st.subheader("🔧 Current Engine Parameters")

st.caption(
    "Enter values within the observed FD001 dataset ranges."
)


col1, col2, col3 = st.columns(3)


# ============================================================
# ENGINE CYCLE
# ============================================================

with col1:

    cycle = st.number_input(
        "Engine Cycle",
        min_value=1.0,
        max_value=362.0,
        value=104.0,
        step=1.0
    )

    st.caption(
        "Operating cycle: 1 – 362"
    )


# ============================================================
# SENSOR 11
# ============================================================

with col2:

    sensor_11 = st.number_input(
        "HPC Outlet Static Pressure (Sensor 11)",
        min_value=46.850,
        max_value=48.530,
        value=47.510,
        step=0.001
    )

    st.caption(
        "Ps30 | Typical: 47.510 psia"
    )


# ============================================================
# SENSOR 9
# ============================================================

with col3:

    sensor_9 = st.number_input(
        "Physical Core Speed (Sensor 9)",
        min_value=9021.730,
        max_value=9244.590,
        value=9060.660,
        step=0.001
    )

    st.caption(
        "Nc | Typical: 9060.660"
    )


col4, col5, col6 = st.columns(3)


# ============================================================
# SENSOR 4
# ============================================================

with col4:

    sensor_4 = st.number_input(
        "LPT Outlet Temperature (Sensor 4)",
        min_value=1382.250,
        max_value=1441.490,
        value=1408.040,
        step=0.001
    )

    st.caption(
        "T50 | Typical: 1408.040 °R"
    )


# ============================================================
# SENSOR 14
# ============================================================

with col5:

    sensor_14 = st.number_input(
        "Corrected Core Speed (Sensor 14)",
        min_value=8099.940,
        max_value=8293.720,
        value=8140.540,
        step=0.001
    )

    st.caption(
        "NRc | Typical: 8140.540"
    )


# ============================================================
# SENSOR 12
# ============================================================

with col6:

    sensor_12 = st.number_input(
        "Fuel Flow / Pressure Ratio (Sensor 12)",
        min_value=518.690,
        max_value=523.380,
        value=521.480,
        step=0.001
    )

    st.caption(
        "φ | Typical: 521.480"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🚀 Analyze Engine Condition",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = default_values.copy()

    input_data["cycle"] = cycle
    input_data["sensor_11"] = sensor_11
    input_data["sensor_9"] = sensor_9
    input_data["sensor_4"] = sensor_4
    input_data["sensor_14"] = sensor_14
    input_data["sensor_12"] = sensor_12

    input_df = pd.DataFrame([input_data])

    input_df = input_df[features]


    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    tree_predictions = np.array([
        tree.predict(input_df)[0]
        for tree in model.estimators_
    ])

    predicted_rul = np.mean(tree_predictions)

    predicted_rul = max(
        0,
        predicted_rul
    )

    rul_cycles = round(predicted_rul)


    # ========================================================
    # PREDICTION UNCERTAINTY
    # ========================================================

    lower_bound = max(
        0,
        np.percentile(tree_predictions, 10)
    )

    upper_bound = max(
        0,
        np.percentile(tree_predictions, 90)
    )

    uncertainty_range = (
        f"{round(lower_bound)} – "
        f"{round(upper_bound)} cycles"
    )


    # ========================================================
    # ENGINE CONDITION
    # ========================================================

    if predicted_rul > 50:

        condition = "Healthy"
        risk = "Low"
        priority = "Routine"

        icon = "🟢"

        recommendation = (
            "Continue routine monitoring and follow "
            "the scheduled maintenance plan."
        )

        status_message = (
            "The predicted remaining life is relatively high. "
            "No immediate maintenance action is required."
        )

        progress_value = 100

        risk_score = max(
            0,
            min(
                round(
                    100 - (predicted_rul / 150) * 100
                ),
                35
            )
        )


    elif predicted_rul >= 20:

        condition = "Warning"
        risk = "Medium"
        priority = "Preventive"

        icon = "🟡"

        recommendation = (
            "Increase monitoring frequency and schedule "
            "preventive maintenance."
        )

        status_message = (
            "The engine has a moderate remaining useful life. "
            "Preventive maintenance should be planned."
        )

        progress_value = int(
            (predicted_rul / 50) * 100
        )

        risk_score = round(
            35 + ((50 - predicted_rul) / 30) * 30
        )


    else:

        condition = "Critical"
        risk = "High"
        priority = "Immediate"

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

        risk_score = round(
            65 + ((20 - predicted_rul) / 20) * 35
        )


    risk_score = max(
        0,
        min(
            risk_score,
            100
        )
    )

    progress_value = max(
        0,
        min(
            progress_value,
            100
        )
    )


    # ========================================================
    # SENSOR DEVIATION ANALYSIS
    # ========================================================

    sensor_values = {

        "sensor_4": sensor_4,
        "sensor_9": sensor_9,
        "sensor_11": sensor_11,
        "sensor_12": sensor_12,
        "sensor_14": sensor_14

    }


    sensor_results = []


    for sensor, value in sensor_values.items():

        info = sensor_ranges[sensor]

        minimum = info["min"]
        maximum = info["max"]

        median = info["median"]

        range_width = maximum - minimum

        deviation = abs(
            value - median
        ) / range_width * 100


        if deviation < 20:

            sensor_status = "Normal"
            sensor_icon = "🟢"

        elif deviation < 40:

            sensor_status = "Moderate Deviation"
            sensor_icon = "🟡"

        else:

            sensor_status = "High Deviation"
            sensor_icon = "🔴"


        sensor_results.append({

            "Sensor": info["name"],
            "Reading": value,
            "Status": f"{sensor_icon} {sensor_status}"

        })


    sensor_df = pd.DataFrame(
        sensor_results
    )


    # ========================================================
    # ENGINE MONITORING DASHBOARD
    # ========================================================

    st.markdown("---")

    st.title("📊 Engine Health Dashboard")

    st.write(
        "AI-based assessment of the current engine operating condition."
    )


    # ========================================================
    # MAIN KPI CARDS
    # ========================================================

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)


    with kpi1:

        st.metric(
            "✈️ Remaining Useful Life",
            f"{rul_cycles} cycles"
        )


    with kpi2:

        st.metric(
            "Engine Condition",
            f"{icon} {condition}"
        )


    with kpi3:

        st.metric(
            "⚠️ Risk Score",
            f"{risk_score}%"
        )


    with kpi4:

        st.metric(
            "🛠️ Maintenance Priority",
            priority
        )


    # ========================================================
    # RUL VISUALIZATION
    # ========================================================

    st.markdown("---")

    st.subheader("🔋 Remaining Engine Life")

    st.progress(
        progress_value
    )

    st.caption(
        f"Predicted RUL: {rul_cycles} cycles | "
        f"Estimated model range: {uncertainty_range}"
    )


    # ========================================================
    # PREDICTION CONFIDENCE
    # ========================================================

    st.markdown("---")

    st.subheader("🎯 Prediction Uncertainty")

    uncertainty_col1, uncertainty_col2 = st.columns(2)


    with uncertainty_col1:

        st.metric(
            "Predicted RUL",
            f"{rul_cycles} cycles"
        )


    with uncertainty_col2:

        st.metric(
            "Estimated Prediction Range",
            uncertainty_range
        )


    st.caption(
        "The prediction range represents the variation "
        "among the individual Random Forest trees and "
        "should be interpreted as model uncertainty, "
        "not a formal statistical confidence interval."
    )


    # ========================================================
    # ENGINE STATUS
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
    # SENSOR MONITORING
    # ========================================================

    st.markdown("---")

    st.subheader("🔬 Sensor Condition Monitoring")

    st.write(
        "Current sensor readings are compared with "
        "their observed operating ranges in the FD001 dataset."
    )


    st.dataframe(
        sensor_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MAINTENANCE RECOMMENDATION
    # ========================================================

    st.markdown("---")

    st.subheader("🛠️ Maintenance Recommendation")

    st.info(
        f"**Priority: {priority}**\n\n"
        f"{recommendation}"
    )


    # ========================================================
    # CURRENT PARAMETERS
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
            "LPT Outlet Temperature",
            f"{sensor_4:.3f}"
        )


    with p2:

        st.metric(
            "Physical Core Speed",
            f"{sensor_9:.3f}"
        )

        st.metric(
            "HPC Outlet Pressure",
            f"{sensor_11:.3f}"
        )


    with p3:

        st.metric(
            "Fuel Flow / Pressure Ratio",
            f"{sensor_12:.3f}"
        )

        st.metric(
            "Corrected Core Speed",
            f"{sensor_14:.3f}"
        )


    # ========================================================
    # DECISION SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader("📝 AI Monitoring Summary")


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            f"**Predicted RUL:** "
            f"{rul_cycles} cycles"
        )

        st.write(
            f"**Engine Condition:** "
            f"{icon} {condition}"
        )

        st.write(
            f"**Risk Level:** "
            f"{risk}"
        )

        st.write(
            f"**Risk Score:** "
            f"{risk_score}%"
        )


    with summary_col2:

        st.write(
            "**Recommended Action:**"
        )

        st.write(
            recommendation
        )

        st.write(
            f"**Maintenance Priority:** "
            f"{priority}"
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
    "and Predictive Maintenance System"
)
