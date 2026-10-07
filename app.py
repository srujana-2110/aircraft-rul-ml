import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aircraft Engine RUL Monitor",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND FEATURES
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

    st.subheader("📌 About")

    st.write(
        "This application predicts the Remaining Useful Life "
        "(RUL) of an aircraft engine using Machine Learning."
    )

    st.markdown("---")

    st.subheader("🤖 Model")

    st.write("Random Forest Regressor")
    st.write("Model Features: 18")
    st.write("User Inputs: 6")

    st.markdown("---")

    st.subheader("📊 Dataset")

    st.write("NASA C-MAPSS FD001")
    st.write("100 training engines")
    st.write("1 operating condition")

    st.markdown("---")

    st.subheader("📈 Dashboard")

    st.write("RUL Analysis")
    st.write("Feature Importance")
    st.write("Prediction Performance")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("✈️ Aircraft Engine RUL Prediction")

st.subheader(
    "Predictive Maintenance & Engine Condition Monitoring"
)

st.write(
    "Enter the current engine parameters to estimate "
    "the Remaining Useful Life (RUL)."
)


# ============================================================
# ENGINE INPUT SECTION
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
        "Range: 1 – 362 | Typical value: 104"
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
        "Range: 46.850 – 48.530 | Typical value: 47.510"
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
        "Range: 9021.730 – 9244.590 | Typical value: 9060.660"
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
        "Range: 1382.250 – 1441.490 | Typical value: 1408.040"
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
        "Range: 8099.940 – 8293.720 | Typical value: 8140.540"
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
        "Range: 518.690 – 523.380 | Typical value: 521.480"
    )


# ============================================================
# SENSOR INFORMATION
# ============================================================

st.info(
    "ℹ️ Sensor values are dataset-specific indicators from "
    "NASA C-MAPSS. They are anonymized and should not be "
    "interpreted as physical PSI, °C, bar, or other engineering units."
)


# ============================================================
# PREDICTION BUTTON
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
    # MODEL PREDICTION
    # --------------------------------------------------------

    predicted_rul = model.predict(input_df)[0]

    predicted_rul = max(0, predicted_rul)

    rul_cycles = round(predicted_rul)


    # --------------------------------------------------------
    # CONDITION CLASSIFICATION
    # --------------------------------------------------------

    if predicted_rul > 50:

        condition = "Healthy"
        risk = "Low"
        icon = "🟢"

        recommendation = (
            "Continue routine monitoring and follow the "
            "scheduled maintenance plan."
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

        progress_value = int(
            (predicted_rul / 20) * 100
        )


    progress_value = max(
        0,
        min(progress_value, 100)
    )


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.markdown("---")

    st.subheader("📊 Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Remaining Useful Life",
            f"{rul_cycles} cycles"
        )


    with result_col2:

        st.metric(
            "Engine Condition",
            f"{icon} {condition}"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            risk
        )


    # ========================================================
    # RUL VISUALIZATION
    # ========================================================

    st.markdown("### 🔋 Engine Life Status")

    st.progress(progress_value)

    st.caption(
        f"Estimated remaining life: {rul_cycles} cycles"
    )


    # ========================================================
    # MAINTENANCE RECOMMENDATION
    # ========================================================

    st.markdown("### 🛠️ Maintenance Recommendation")

    if condition == "Healthy":

        st.success(
            f"🟢 **Healthy Condition**\n\n"
            f"{recommendation}"
        )

    elif condition == "Warning":

        st.warning(
            f"🟡 **Warning Condition**\n\n"
            f"{recommendation}"
        )

    else:

        st.error(
            f"🔴 **Critical Condition**\n\n"
            f"{recommendation}"
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader("📋 Input Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)


    with summary_col1:

        st.write(
            f"**Engine Cycle:** {cycle:.0f}"
        )

        st.write(
            f"**Thermal Indicator:** {sensor_11:.3f}"
        )


    with summary_col2:

        st.write(
            f"**Pressure/Performance Indicator:** "
            f"{sensor_9:.3f}"
        )

        st.write(
            f"**Engine Health Indicator:** "
            f"{sensor_4:.3f}"
        )


    with summary_col3:

        st.write(
            f"**Performance Indicator:** "
            f"{sensor_14:.3f}"
        )

        st.write(
            f"**Condition Indicator:** "
            f"{sensor_12:.3f}"
        )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

st.markdown("---")

st.title("📈 Analytics Dashboard")

st.write(
    "Model analysis and performance visualizations "
    "based on the project dataset and trained Random Forest model."
)


# ============================================================
# LOAD PREDICTION DATA
# ============================================================

prediction_file = "aircraft_rul_predictions.csv"

if os.path.exists(prediction_file):

    predictions_df = pd.read_csv(
        prediction_file
    )

else:

    predictions_df = None


# ============================================================
# DASHBOARD METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Model",
        "Random Forest"
    )


with metric2:

    st.metric(
        "Model Features",
        "18"
    )


with metric3:

    st.metric(
        "User Inputs",
        "6"
    )


with metric4:

    st.metric(
        "Dataset",
        "C-MAPSS FD001"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown("---")

st.subheader("🎯 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

fig1, ax1 = plt.subplots(
    figsize=(9, 6)
)

ax1.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax1.set_xlabel(
    "Importance"
)

ax1.set_ylabel(
    "Feature"
)

ax1.set_title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

st.pyplot(fig1)

plt.close(fig1)


# ============================================================
# ACTUAL VS PREDICTED RUL
# ============================================================

if predictions_df is not None:

    st.markdown("---")

    st.subheader("🎯 Actual vs Predicted RUL")

    # Try to identify the correct columns
    actual_column = None
    predicted_column = None

    for col in predictions_df.columns:

        col_lower = col.lower()

        if (
            "actual" in col_lower
            and "rul" in col_lower
        ):
            actual_column = col

        if (
            "predicted" in col_lower
            and "rul" in col_lower
        ):
            predicted_column = col


    if actual_column and predicted_column:

        fig2, ax2 = plt.subplots(
            figsize=(9, 6)
        )

        ax2.scatter(
            predictions_df[actual_column],
            predictions_df[predicted_column],
            alpha=0.5
        )

        min_value = min(
            predictions_df[actual_column].min(),
            predictions_df[predicted_column].min()
        )

        max_value = max(
            predictions_df[actual_column].max(),
            predictions_df[predicted_column].max()
        )

        ax2.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        ax2.set_xlabel(
            "Actual RUL"
        )

        ax2.set_ylabel(
            "Predicted RUL"
        )

        ax2.set_title(
            "Actual vs Predicted Remaining Useful Life"
        )

        plt.tight_layout()

        st.pyplot(fig2)

        plt.close(fig2)

    else:

        st.warning(
            "Actual and predicted RUL columns were not "
            "identified in aircraft_rul_predictions.csv."
        )


# ============================================================
# RUL DISTRIBUTION
# ============================================================

if predictions_df is not None:

    st.markdown("---")

    st.subheader("📊 RUL Distribution")

    actual_column = None

    for col in predictions_df.columns:

        if (
            "actual" in col.lower()
            and "rul" in col.lower()
        ):

            actual_column = col
            break


    if actual_column:

        fig3, ax3 = plt.subplots(
            figsize=(9, 5)
        )

        ax3.hist(
            predictions_df[actual_column].dropna(),
            bins=30
        )

        ax3.set_xlabel(
            "Remaining Useful Life (Cycles)"
        )

        ax3.set_ylabel(
            "Number of Engines"
        )

        ax3.set_title(
            "Distribution of Actual RUL"
        )

        plt.tight_layout()

        st.pyplot(fig3)

        plt.close(fig3)


# ============================================================
# CONDITION DISTRIBUTION
# ============================================================

if predictions_df is not None:

    st.markdown("---")

    st.subheader("🟢🟡🔴 Engine Condition Distribution")

    predicted_column = None

    for col in predictions_df.columns:

        if (
            "predicted" in col.lower()
            and "rul" in col.lower()
        ):

            predicted_column = col
            break


    if predicted_column:

        condition_data = predictions_df[
            predicted_column
        ].copy()

        condition_data = condition_data.clip(
            lower=0
        )


        def classify_condition(rul):

            if rul > 50:
                return "Healthy"

            elif rul >= 20:
                return "Warning"

            else:
                return "Critical"


        condition_counts = (
            condition_data
            .apply(classify_condition)
            .value_counts()
        )


        fig4, ax4 = plt.subplots(
            figsize=(8, 5)
        )

        ax4.bar(
            condition_counts.index,
            condition_counts.values
        )

        ax4.set_xlabel(
            "Engine Condition"
        )

        ax4.set_ylabel(
            "Number of Predictions"
        )

        ax4.set_title(
            "Predicted Engine Condition Distribution"
        )

        plt.tight_layout()

        st.pyplot(fig4)

        plt.close(fig4)


# ============================================================
# PROJECT INSIGHTS
# ============================================================

st.markdown("---")

st.subheader("💡 Project Insights")

insight_col1, insight_col2 = st.columns(2)


with insight_col1:

    st.markdown(
        """
        **Key Model Insights**

        - Random Forest is used for RUL regression.
        - 18 features are used by the trained model.
        - Engine cycle is an important indicator of engine aging.
        - Sensor measurements provide additional information
          about engine condition.
        """
    )


with insight_col2:

    st.markdown(
        """
        **Maintenance Strategy**

        - 🟢 **Healthy:** Continue routine monitoring.
        - 🟡 **Warning:** Increase monitoring and plan
          preventive maintenance.
        - 🔴 **Critical:** Prioritize inspection and
          maintenance planning.
        """
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
    "⚠️ This application is based on the NASA C-MAPSS "
    "simulated dataset and is intended for educational "
    "and project demonstration purposes."
)
