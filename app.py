import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    page_icon="âœˆï¸",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    predicted_rul,
    rul_cycles,
    prediction_range,
    condition,
    risk,
    risk_score,
    priority,
    recommendation,
    cycle,
    sensor_4,
    sensor_9,
    sensor_11,
    sensor_12,
    sensor_14
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=12
    )

    story = []

    story.append(
        Paragraph(
            "Aircraft Engine Predictive Maintenance Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Remaining Useful Life Prediction & Condition Monitoring",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 12))

    summary_data = [
        ["Prediction Summary", "Value"],
        ["Predicted RUL", f"{rul_cycles} cycles"],
        ["Model Prediction Range", prediction_range],
        ["Engine Condition", condition],
        ["Risk Level", risk],
        ["Risk Score", f"{risk_score}%"],
        ["Maintenance Priority", priority]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[220, 220]
    )

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(summary_table)
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "Current Engine Parameters",
            styles["Heading2"]
        )
    )

    parameter_data = [
        ["Parameter", "Current Value", "Typical Value"],
        ["Engine Cycle", f"{cycle:.0f}", "104.000"],
        ["LPT Outlet Temperature (Sensor 4)", f"{sensor_4:.3f}", "1408.040"],
        ["Physical Core Speed (Sensor 9)", f"{sensor_9:.3f}", "9060.660"],
        ["HPC Outlet Static Pressure (Sensor 11)", f"{sensor_11:.3f}", "47.510"],
        ["Fuel Flow / Pressure Ratio (Sensor 12)", f"{sensor_12:.3f}", "521.480"],
        ["Corrected Core Speed (Sensor 14)", f"{sensor_14:.3f}", "8140.540"]
    ]

    parameter_table = Table(
        parameter_data,
        colWidths=[250, 100, 100]
    )

    parameter_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(parameter_table)
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "Maintenance Recommendation",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Priority:</b> {priority}<br/>"
            f"{recommendation}",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "Model Information",
            styles["Heading2"]
        )
    )

    model_data = [
        ["Item", "Details"],
        ["Dataset", "NASA C-MAPSS FD001"],
        ["Algorithm", "Random Forest Regressor"],
        ["Model Features", "18"],
        ["Decision Trees", "50"],
        ["Validation MAE", "23.78 cycles"],
        ["Validation RMSE", "31.31 cycles"],
        ["Validation RÂ²", "0.7725"]
    ]

    model_table = Table(
        model_data,
        colWidths=[180, 270]
    )

    model_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#70AD47")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(model_table)
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "Note: The model prediction range represents variation among "
            "individual Random Forest tree predictions and is not a formal "
            "statistical confidence interval. This application is a project "
            "demonstration based on the simulated NASA C-MAPSS FD001 dataset "
            "and should not be treated as a certified aviation maintenance decision system.",
            styles["BodyText"]
        )
    )

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "aircraft_rul_random_forest.pkl"
    )

    features = joblib.load(
        "rul_features.pkl"
    )

    return model, features


model, features = load_model()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_training_data(file_path):

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=DATA_COLUMNS
    )

    return df


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
# SENSOR INFORMATION
# ============================================================

sensor_ranges = {

    "sensor_4": {

        "name": "LPT Outlet Temperature",

        "symbol": "T50",

        "min": 1382.250,

        "max": 1441.490,

        "median": 1408.040

    },

    "sensor_9": {

        "name": "Physical Core Speed",

        "symbol": "Nc",

        "min": 9021.730,

        "max": 9244.590,

        "median": 9060.660

    },

    "sensor_11": {

        "name": "HPC Outlet Static Pressure",

        "symbol": "Ps30",

        "min": 46.850,

        "max": 48.530,

        "median": 47.510

    },

    "sensor_12": {

        "name": "Fuel Flow / Pressure Ratio",

        "symbol": "Ï†",

        "min": 518.690,

        "max": 523.380,

        "median": 521.480

    },

    "sensor_14": {

        "name": "Corrected Core Speed",

        "symbol": "NRc",

        "min": 8099.940,

        "max": 8293.720,

        "median": 8140.540

    }
}


# ============================================================
# LOAD DATASET
# ============================================================

dataset = None

if os.path.exists("train_FD001.txt"):

    try:

        dataset = load_training_data(
            "train_FD001.txt"
        )

    except Exception:
        dataset = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("âœˆï¸ Aircraft RUL Monitor")

    st.markdown("---")

    st.subheader("ðŸ“Œ Project")

    st.write(
        "Aircraft Engine Remaining Useful Life "
        "(RUL) Prediction and Condition Monitoring "
        "using Machine Learning."
    )

    st.markdown("---")

    st.subheader("ðŸ¤– Model")

    st.write(
        "Random Forest Regressor"
    )

    st.write(
        "18 model features"
    )

    st.write(
        "50 decision trees"
    )

    st.markdown("---")

    st.subheader("ðŸ“Š Dataset")

    st.write(
        "NASA C-MAPSS FD001"
    )

    st.write(
        "100 training engines"
    )

    st.write(
        "20,631 training records"
    )

    st.markdown("---")

    st.subheader("âš™ï¸ System")

    st.write("âœ“ RUL Prediction")

    st.write("âœ“ Risk Assessment")

    st.write("âœ“ Sensor Monitoring")

    st.write("âœ“ Dataset Explorer")

    st.write("âœ“ Maintenance Recommendation")


# ============================================================
# MAIN NAVIGATION
# ============================================================

st.title(
    "âœˆï¸ Aircraft Engine Predictive Maintenance"
)

st.caption(
    "Remaining Useful Life Prediction & Condition Monitoring"
)

page = st.radio(
    "Navigation",
    [
        "ðŸ  Prediction Dashboard",
        "ðŸ“‚ Dataset Explorer",
        "ðŸ¤– Model Information"
    ],
    horizontal=True
)


# ============================================================
# ============================================================
# PAGE 1 â€” PREDICTION DASHBOARD
# ============================================================
# ============================================================

if page == "ðŸ  Prediction Dashboard":

    st.subheader(
        "ðŸ”§ Current Engine Parameters"
    )

    st.write(
        "Enter the current engine operating parameters "
        "to estimate remaining useful life and assess "
        "maintenance condition."
    )

    st.caption(
        "Input values are restricted to the observed "
        "ranges in the FD001 training dataset."
    )


    # ========================================================
    # INPUT ROW 1
    # ========================================================

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # ENGINE CYCLE
    # --------------------------------------------------------

    with col1:

        cycle = st.number_input(

            "Engine Cycle",

            min_value=0.0,

            max_value=500.0,

            value=104.0,

            step=1.0

        )

        st.caption(
            "Range: 1 â€“ 362 | Typical: 104"
        )


    # --------------------------------------------------------
    # SENSOR 11
    # --------------------------------------------------------

    with col2:

        sensor_11 = st.number_input(

            "HPC Outlet Static Pressure (Sensor 11)",

            min_value=0.0,

            max_value=100.0,

            value=47.510,

            step=0.001

        )

        st.caption(
            "Range: 46.850 â€“ 48.530 | Typical: 47.510"
        )


    # --------------------------------------------------------
    # SENSOR 9
    # --------------------------------------------------------

    with col3:

        sensor_9 = st.number_input(

            "Physical Core Speed (Sensor 9)",

            min_value=0.0,

            max_value=10000.0,

            value=9060.660,

            step=0.001

        )

        st.caption(
            "Range: 9021.730 â€“ 9244.590 | Typical: 9060.660"
        )


    # ========================================================
    # INPUT ROW 2
    # ========================================================

    col4, col5, col6 = st.columns(3)


    # --------------------------------------------------------
    # SENSOR 4
    # --------------------------------------------------------

    with col4:

        sensor_4 = st.number_input(

            "LPT Outlet Temperature (Sensor 4)",

            min_value=0.0,

            max_value=2000.0,

            value=1408.040,

            step=0.001

        )

        st.caption(
            "Range: 1382.250 â€“ 1441.490 | Typical: 1408.040"
        )


    # --------------------------------------------------------
    # SENSOR 14
    # --------------------------------------------------------

    with col5:

        sensor_14 = st.number_input(

            "Corrected Core Speed (Sensor 14)",

            min_value=0.0,

            max_value=10000.0,

            value=8140.540,

            step=0.001

        )

        st.caption(
            "Range: 8099.940 â€“ 8293.720 | Typical: 8140.540"
        )


    # --------------------------------------------------------
    # SENSOR 12
    # --------------------------------------------------------

    with col6:

        sensor_12 = st.number_input(

            "Fuel Flow / Pressure Ratio (Sensor 12)",

            min_value=0.0,

            max_value=1000.0,

            value=521.480,

            step=0.001

        )

        st.caption(
            "Range: 518.690 â€“ 523.380 | Typical: 521.480"
        )


    st.info(
        "Input boxes allow broader values so the app can validate them. "
        "Prediction is blocked if any value falls outside the observed "
        "FD001 training-data range shown below each field."
    )

    # ========================================================
    # INPUT VALIDATION RANGES
    # ========================================================

    input_ranges = {
        "Engine Cycle": (1.0, 362.0),
        "HPC Outlet Static Pressure (Sensor 11)": (46.850, 48.530),
        "Physical Core Speed (Sensor 9)": (9021.730, 9244.590),
        "LPT Outlet Temperature (Sensor 4)": (1382.250, 1441.490),
        "Corrected Core Speed (Sensor 14)": (8099.940, 8293.720),
        "Fuel Flow / Pressure Ratio (Sensor 12)": (518.690, 523.380)
    }

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown("---")

    predict_button = st.button(

        "ðŸš€ Analyze Engine Condition",

        use_container_width=True

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        # ====================================================
        # INPUT VALIDATION
        # ====================================================

        user_inputs = {
            "Engine Cycle": cycle,
            "HPC Outlet Static Pressure (Sensor 11)": sensor_11,
            "Physical Core Speed (Sensor 9)": sensor_9,
            "LPT Outlet Temperature (Sensor 4)": sensor_4,
            "Corrected Core Speed (Sensor 14)": sensor_14,
            "Fuel Flow / Pressure Ratio (Sensor 12)": sensor_12
        }

        invalid_inputs = []
        boundary_warnings = []

        for parameter, value in user_inputs.items():
            minimum, maximum = input_ranges[parameter]

            if value < minimum or value > maximum:
                invalid_inputs.append(
                    f"ðŸ”´ **{parameter}** = `{value:.3f}` is outside "
                    f"the observed training range `{minimum:.3f} â€“ {maximum:.3f}`."
                )
            else:
                boundary = (maximum - minimum) * 0.05
                if value <= minimum + boundary or value >= maximum - boundary:
                    boundary_warnings.append(
                        f"âš ï¸ **{parameter}** = `{value:.3f}` is close to "
                        "the boundary of the observed training range."
                    )

        if invalid_inputs:
            st.error("ðŸš¨ Invalid Parameter Values")
            st.warning(
                "One or more entered values are outside the ranges observed "
                "in the model's training data. The RUL prediction has been "
                "blocked because the model may be unreliable outside its "
                "training range."
            )
            for message in invalid_inputs:
                st.markdown(message)
            st.info(
                "ðŸ’¡ Please enter values within the displayed observed ranges "
                "and analyze the engine again."
            )
            st.stop()

        if boundary_warnings:
            st.warning("âš ï¸ Input Boundary Warning")
            for message in boundary_warnings:
                st.markdown(message)
            st.caption(
                "The prediction is still allowed, but values close to the "
                "training-data boundaries may be less reliable."
            )

        # ====================================================
        # CREATE MODEL INPUT
        # ====================================================

        input_data = default_values.copy()

        input_data["cycle"] = cycle

        input_data["sensor_11"] = sensor_11

        input_data["sensor_9"] = sensor_9

        input_data["sensor_4"] = sensor_4

        input_data["sensor_14"] = sensor_14

        input_data["sensor_12"] = sensor_12


        input_df = pd.DataFrame(
            [input_data]
        )

        input_df = input_df[
            features
        ]


        # ====================================================
        # RANDOM FOREST PREDICTIONS
        # ====================================================

        tree_predictions = np.array([

            tree.predict(input_df)[0]

            for tree in model.estimators_

        ])


        predicted_rul = np.mean(
            tree_predictions
        )

        predicted_rul = max(
            0,
            predicted_rul
        )

        rul_cycles = round(
            predicted_rul
        )


        # ====================================================
        # MODEL PREDICTION RANGE
        # ====================================================

        lower_bound = max(

            0,

            np.percentile(
                tree_predictions,
                10
            )

        )

        upper_bound = max(

            0,

            np.percentile(
                tree_predictions,
                90
            )

        )

        prediction_range = (

            f"{round(lower_bound)} â€“ "
            f"{round(upper_bound)} cycles"

        )


        # ====================================================
        # ENGINE CONDITION
        # ====================================================

        if predicted_rul > 50:

            condition = "Healthy"

            risk = "Low"

            priority = "Routine"

            icon = "ðŸŸ¢"

            recommendation = (

                "Continue routine monitoring and "
                "follow the scheduled maintenance plan."

            )

            status_message = (

                "The predicted remaining life is relatively "
                "high. No immediate maintenance action is required."

            )

            progress_value = 100

            risk_score = round(

                min(

                    35,

                    max(

                        0,

                        100 -
                        (predicted_rul / 150) * 100

                    )

                )

            )


        elif predicted_rul >= 20:

            condition = "Warning"

            risk = "Medium"

            priority = "Preventive"

            icon = "ðŸŸ¡"

            recommendation = (

                "Increase monitoring frequency and "
                "schedule preventive maintenance."

            )

            status_message = (

                "The engine has a moderate remaining "
                "useful life. Preventive maintenance "
                "should be planned."

            )

            progress_value = int(

                (predicted_rul / 50) * 100

            )

            risk_score = round(

                35 +
                ((50 - predicted_rul) / 30) * 30

            )


        else:

            condition = "Critical"

            risk = "High"

            priority = "Immediate"

            icon = "ðŸ”´"

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

                65 +
                ((20 - predicted_rul) / 20) * 35

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


        # ====================================================
        # SENSOR MONITORING
        # ====================================================

        sensor_values = {

            "sensor_4": sensor_4,

            "sensor_9": sensor_9,

            "sensor_11": sensor_11,

            "sensor_12": sensor_12,

            "sensor_14": sensor_14

        }


        sensor_results = []


        for sensor, value in sensor_values.items():

            info = sensor_ranges[
                sensor
            ]

            minimum = info["min"]

            maximum = info["max"]

            median = info["median"]

            range_width = (
                maximum - minimum
            )

            deviation = (

                abs(
                    value - median
                )
                /
                range_width
                *
                100

            )


            if deviation < 20:

                sensor_status = "Normal"

                sensor_icon = "ðŸŸ¢"


            elif deviation < 40:

                sensor_status = "Moderate Deviation"

                sensor_icon = "ðŸŸ¡"


            else:

                sensor_status = "High Deviation"

                sensor_icon = "ðŸ”´"


            sensor_results.append({

                "Sensor": info["name"],

                "Symbol": info["symbol"],

                "Current Reading": round(
                    value,
                    3
                ),

                "Typical Value": round(
                    median,
                    3
                ),

                "Status":
                    f"{sensor_icon} "
                    f"{sensor_status}"

            })


        sensor_df = pd.DataFrame(
            sensor_results
        )


        # ====================================================
        # DASHBOARD
        # ====================================================

        st.markdown("---")

        st.title(
            "ðŸ“Š Engine Monitoring Dashboard"
        )

        st.write(
            "AI-based assessment of the current "
            "engine operating condition."
        )


        # ====================================================
        # KPI CARDS
        # ====================================================

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)


        with kpi1:

            st.metric(

                "âœˆï¸ Predicted RUL",

                f"{rul_cycles} cycles"

            )


        with kpi2:

            st.metric(

                "Engine Condition",

                f"{icon} {condition}"

            )


        with kpi3:

            st.metric(

                "âš ï¸ Risk Score",

                f"{risk_score}%"

            )


        with kpi4:

            st.metric(

                "ðŸ› ï¸ Maintenance Priority",

                priority

            )


        # ====================================================
        # RUL VISUAL
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ”‹ Remaining Engine Life"
        )

        st.progress(
            progress_value
        )

        st.caption(

            f"Predicted remaining useful life: "
            f"{rul_cycles} cycles"

        )


        # ====================================================
        # MODEL RANGE
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸŽ¯ Model Prediction Range"
        )

        range_col1, range_col2 = st.columns(2)


        with range_col1:

            st.metric(

                "Predicted RUL",

                f"{rul_cycles} cycles"

            )


        with range_col2:

            st.metric(

                "Model Prediction Range",

                prediction_range

            )


        st.caption(

            "The prediction range represents the variation "
            "among individual Random Forest tree predictions. "
            "It is not a formal statistical confidence interval."

        )


        # ====================================================
        # ENGINE STATUS
        # ====================================================

        st.markdown("---")

        st.subheader(

            f"{icon} Engine Status: "
            f"{condition}"

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


        # ====================================================
        # SENSOR MONITORING
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ”¬ Sensor Condition Monitoring"
        )

        st.write(

            "Current sensor readings are compared "
            "with their observed operating ranges "
            "in the FD001 training dataset."

        )

        st.dataframe(

            sensor_df,

            use_container_width=True,

            hide_index=True

        )


        # ====================================================
        # MAINTENANCE
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ› ï¸ Maintenance Recommendation"
        )

        st.info(

            f"**Maintenance Priority: "
            f"{priority}**\n\n"
            f"{recommendation}"

        )


        # ====================================================
        # CURRENT PARAMETERS
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ“‹ Current Engine Parameters"
        )


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


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ“ AI Monitoring Summary"
        )


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

        # ====================================================
        # PDF REPORT
        # ====================================================

        st.markdown("---")

        pdf_data = create_pdf_report(
            predicted_rul=predicted_rul,
            rul_cycles=rul_cycles,
            prediction_range=prediction_range,
            condition=condition,
            risk=risk,
            risk_score=risk_score,
            priority=priority,
            recommendation=recommendation,
            cycle=cycle,
            sensor_4=sensor_4,
            sensor_9=sensor_9,
            sensor_11=sensor_11,
            sensor_12=sensor_12,
            sensor_14=sensor_14
        )

        st.download_button(
            "ðŸ“¥ Download PDF Report",
            data=pdf_data,
            file_name="aircraft_engine_rul_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# ============================================================
# PAGE 2 â€” DATASET EXPLORER
# ============================================================
# ============================================================

elif page == "ðŸ“‚ Dataset Explorer":

    st.header(
        "ðŸ“‚ NASA C-MAPSS FD001 Dataset Explorer"
    )

    st.write(
        "Explore the training data used to develop "
        "the aircraft engine RUL prediction model."
    )


    # ========================================================
    # DATASET CHECK
    # ========================================================

    if dataset is None:

        st.warning(
            "train_FD001.txt was not found in the project folder."
        )

        st.info(
            "Add train_FD001.txt to the same folder as app.py "
            "and restart the application."
        )

        uploaded_file = st.file_uploader(

            "Or upload train_FD001.txt",

            type=["txt"]

        )


        if uploaded_file is not None:

            dataset = pd.read_csv(

                uploaded_file,

                sep=r"\s+",

                header=None,

                names=DATA_COLUMNS

            )


    # ========================================================
    # DISPLAY DATA
    # ========================================================

    if dataset is not None:

        # ====================================================
        # CALCULATE RUL
        # ====================================================

        max_cycles = (

            dataset
            .groupby("unit_id")["cycle"]
            .max()
            .reset_index()

        )

        max_cycles.rename(

            columns={
                "cycle": "max_cycle"
            },

            inplace=True

        )


        dataset_with_rul = dataset.merge(

            max_cycles,

            on="unit_id",

            how="left"

        )


        dataset_with_rul["RUL"] = (

            dataset_with_rul["max_cycle"]
            -
            dataset_with_rul["cycle"]

        )


        # ====================================================
        # DATASET KPIs
        # ====================================================

        total_records = len(
            dataset
        )

        total_engines = dataset[
            "unit_id"
        ].nunique()

        total_features = len(
            dataset.columns
        )

        max_cycle = int(
            dataset["cycle"].max()
        )


        d1, d2, d3, d4 = st.columns(4)


        with d1:

            st.metric(
                "Training Records",
                f"{total_records:,}"
            )


        with d2:

            st.metric(
                "Training Engines",
                total_engines
            )


        with d3:

            st.metric(
                "Dataset Columns",
                total_features
            )


        with d4:

            st.metric(
                "Maximum Cycle",
                max_cycle
            )


        # ====================================================
        # DATASET QUALITY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ”Ž Dataset Quality"
        )


        missing_values = int(
            dataset.isnull()
            .sum()
            .sum()
        )

        duplicate_rows = int(
            dataset.duplicated()
            .sum()
        )


        q1, q2, q3 = st.columns(3)


        with q1:

            st.metric(
                "Missing Values",
                missing_values
            )


        with q2:

            st.metric(
                "Duplicate Rows",
                duplicate_rows
            )


        with q3:

            st.metric(
                "Unique Engines",
                total_engines
            )


        # ====================================================
        # DATA PREVIEW
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ“„ Dataset Preview"
        )

        preview_rows = st.slider(

            "Number of rows to display",

            min_value=5,

            max_value=50,

            value=10,

            step=5

        )


        st.dataframe(

            dataset_with_rul.head(
                preview_rows
            ),

            use_container_width=True,

            hide_index=True

        )


        # ====================================================
        # ENGINE EXPLORER
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ” Explore Individual Engine"
        )

        selected_engine = st.selectbox(

            "Select Engine ID",

            sorted(
                dataset["unit_id"]
                .unique()
            )

        )


        engine_data = (

            dataset_with_rul[
                dataset_with_rul["unit_id"]
                == selected_engine
            ]

        )


        engine_max_cycle = int(
            engine_data["cycle"].max()
        )

        engine_current_rul = int(
            engine_data["RUL"].iloc[0]
        )


        e1, e2, e3 = st.columns(3)


        with e1:

            st.metric(
                "Engine ID",
                selected_engine
            )


        with e2:

            st.metric(
                "Total Cycles",
                engine_max_cycle
            )


        with e3:

            st.metric(
                "Initial RUL",
                engine_current_rul
            )


        st.write(
            f"**Engine {selected_engine} cycle history**"
        )


        engine_display = engine_data[

            [
                "unit_id",
                "cycle",
                "sensor_4",
                "sensor_9",
                "sensor_11",
                "sensor_12",
                "sensor_14",
                "RUL"
            ]

        ]


        st.dataframe(

            engine_display,

            use_container_width=True,

            hide_index=True

        )


        # ====================================================
        # RUL EXPLANATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ§® How RUL Is Calculated"
        )

        st.code(
            "RUL = Maximum Cycle of Engine - Current Cycle"
        )

        st.write(
            "For example, if an engine's maximum training "
            "cycle is 200 and its current cycle is 150:"
        )

        st.info(
            "RUL = 200 âˆ’ 150 = 50 cycles"
        )


        # ====================================================
        # COLUMN INFORMATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "ðŸ“‹ Dataset Columns"
        )


        column_info = pd.DataFrame({

            "Column": DATA_COLUMNS,

            "Description": [

                "Engine identifier",

                "Operating cycle",

                "Operational setting 1",

                "Operational setting 2",

                "Operational setting 3",

                "Sensor 1",

                "Sensor 2",

                "Sensor 3",

                "LPT outlet temperature",

                "Sensor 5",

                "Sensor 6",

                "Sensor 7",

                "Sensor 8",

                "Physical core speed",

                "Sensor 10",

                "HPC outlet static pressure",

                "Fuel flow / pressure ratio",

                "Sensor 13",

                "Corrected core speed",

                "Sensor 15",

                "Sensor 16",

                "Sensor 17",

                "Sensor 18",

                "Sensor 19",

                "Sensor 20",

                "Sensor 21"

            ]

        })


        st.dataframe(

            column_info,

            use_container_width=True,

            hide_index=True

        )


# ============================================================
# ============================================================
# PAGE 3 â€” MODEL INFORMATION
# ============================================================
# ============================================================

else:

    st.header(
        "ðŸ¤– Machine Learning Model Information"
    )

    st.write(
        "Technical overview of the preprocessing, "
        "model and evaluation used in the project."
    )


    # ========================================================
    # MODEL OVERVIEW
    # ========================================================

    st.subheader(
        "ðŸŒ³ Model Overview"
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        st.metric(
            "Algorithm",
            "Random Forest"
        )


    with m2:

        st.metric(
            "Model Features",
            "18"
        )


    with m3:

        st.metric(
            "Decision Trees",
            "50"
        )


    st.write(
        "The Random Forest Regressor predicts the remaining "
        "useful life of an aircraft engine based on engine "
        "operating conditions and sensor measurements."
    )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "ðŸ“Š Dataset"
    )


    st.write(
        "**Dataset:** NASA C-MAPSS FD001"
    )

    st.write(
        "**Training Engines:** 100"
    )

    st.write(
        "**Training Records:** 20,631"
    )

    st.write(
        "**Operating Conditions:** 1"
    )

    st.write(
        "**Fault Mode:** 1"
    )


    st.info(
        "C-MAPSS is a simulated turbofan engine dataset. "
        "The sensor measurements represent simulated engine "
        "operating conditions."
    )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    st.markdown("---")

    st.subheader(
        "âš™ï¸ Data Preprocessing"
    )


    preprocessing_steps = [

        "Loaded NASA C-MAPSS FD001 training data.",

        "Assigned meaningful column names.",

        "Calculated training RUL from engine cycle history.",

        "Checked missing values and duplicate records.",

        "Removed constant features with no variation.",

        "Used engine-level train/validation splitting.",

        "Selected 18 useful features for model training."

    ]


    for step in preprocessing_steps:

        st.write(
            f"âœ“ {step}"
        )


    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown("---")

    st.subheader(
        "ðŸ”¢ Model Features"
    )


    feature_df = pd.DataFrame({

        "Feature": features

    })


    st.dataframe(

        feature_df,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "ðŸ“ˆ Model Performance"
    )


    performance_df = pd.DataFrame({

        "Metric": [

            "Validation MAE",

            "Validation RMSE",

            "Validation RÂ²"

        ],

        "Value": [

            "23.78 cycles",

            "31.31 cycles",

            "0.7725"

        ]

    })


    st.dataframe(

        performance_df,

        use_container_width=True,

        hide_index=True

    )


    st.write(
        "Validation results are based on the lightweight "
        "Random Forest model used in the deployed application."
    )


    # ========================================================
    # CONDITION RULES
    # ========================================================

    st.markdown("---")

    st.subheader(
        "ðŸš¦ Engine Condition Rules"
    )


    condition_df = pd.DataFrame({

        "Predicted RUL": [

            "> 50 cycles",

            "20 â€“ 50 cycles",

            "< 20 cycles"

        ],

        "Condition": [

            "ðŸŸ¢ Healthy",

            "ðŸŸ¡ Warning",

            "ðŸ”´ Critical"

        ],

        "Recommended Action": [

            "Routine monitoring",

            "Preventive maintenance",

            "Immediate inspection"

        ]

    })


    st.dataframe(

        condition_df,

        use_container_width=True,

        hide_index=True

    )


    st.caption(
        "These condition thresholds are project-defined "
        "decision rules and are not official NASA labels."
    )


    # ========================================================
    # PROJECT WORKFLOW
    # ========================================================

    st.markdown("---")

    st.subheader(
        "ðŸ”„ Project Workflow"
    )


    st.write(
        "NASA C-MAPSS Dataset"
    )

    st.write(
        "â†“"
    )

    st.write(
        "Data Preprocessing"
    )

    st.write(
        "â†“"
    )

    st.write(
        "Feature Selection"
    )

    st.write(
        "â†“"
    )

    st.write(
        "Random Forest Regression"
    )

    st.write(
        "â†“"
    )

    st.write(
        "RUL Prediction"
    )

    st.write(
        "â†“"
    )

    st.write(
        "Risk & Condition Assessment"
    )

    st.write(
        "â†“"
    )

    st.write(
        "Maintenance Recommendation"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "NASA C-MAPSS FD001 | "
    "Random Forest Regressor | "
    "Aircraft Engine Predictive Maintenance"
)
