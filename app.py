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
    page_icon="✈️",
    layout="wide"
)

# ============================================================
# CUSTOM AVIATION UI
# ============================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(
        135deg,
        #061426 0%,
        #0b1f3a 50%,
        #061426 100%
    );
}

/* Main title */
h1 {
    font-weight: 700;
}

/* Animated airplane */
.airplane {
    position: fixed;
    top: 90px;
    left: -120px;
    font-size: 55px;
    z-index: 999;
    animation: flyAcross 14s linear infinite;
    pointer-events: none;
    opacity: 0.85;
}

@keyframes flyAcross {

    0% {
        left: -120px;
        transform: rotate(-3deg);
    }

    45% {
        left: 45%;
        transform: rotate(-1deg);
    }

    100% {
        left: 110%;
        transform: rotate(-3deg);
    }

}

/* Airplane trail */
.airplane-trail {
    position: fixed;
    top: 125px;
    left: -300px;
    width: 180px;
    height: 3px;
    background: rgba(255,255,255,0.25);
    animation: trailAcross 14s linear infinite;
    pointer-events: none;
}

@keyframes trailAcross {

    0% {
        left: -300px;
        opacity: 0;
    }

    15% {
        opacity: 1;
    }

    100% {
        left: 100%;
        opacity: 0;
    }

}

/* Cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.10);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

</style>

<div class="airplane">✈️</div>
<div class="airplane-trail"></div>

""", unsafe_allow_html=True)
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
    """Create a descriptive PDF report for the latest prediction."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="Aircraft Engine Predictive Maintenance Report",
        author="Aircraft Engine RUL Prediction System"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=14,
        spaceAfter=16
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=7
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555")
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=8
    )

    story = []

    # ------------------------------------------------------------
    # TITLE
    # ------------------------------------------------------------
    story.append(Paragraph("Aircraft Engine Predictive Maintenance Report", title_style))
    story.append(Paragraph(
        "Remaining Useful Life (RUL) Prediction & Condition Monitoring",
        subtitle_style
    ))

    story.append(Paragraph(
        "This report summarizes the latest engine condition assessment generated by the "
        "machine-learning application. It combines the predicted remaining useful life, "
        "model prediction range, condition classification, risk assessment, monitored "
        "sensor readings, and recommended maintenance action.",
        body_style
    ))

    # ------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary", heading_style))

    summary_text = (
        f"The Random Forest model estimates approximately <b>{rul_cycles} cycles</b> of "
        f"remaining useful life for the entered engine condition. The model's tree-level "
        f"prediction range is <b>{prediction_range}</b>. Based on the project's defined "
        f"RUL thresholds, the engine is classified as <b>{condition}</b>, with a "
        f"<b>{risk}</b> risk level and <b>{priority}</b> maintenance priority. "
        f"The current risk score shown in the application is <b>{risk_score}%</b>."
    )
    story.append(Paragraph(summary_text, body_style))

    summary_data = [
        ["Assessment", "Result"],
        ["Predicted RUL", f"{rul_cycles} cycles"],
        ["Model Prediction Range", prediction_range],
        ["Engine Condition", condition],
        ["Risk Level", risk],
        ["Risk Score", f"{risk_score}%"],
        ["Maintenance Priority", priority]
    ]

    summary_table = Table(summary_data, colWidths=[220, 220], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # ------------------------------------------------------------
    # RUL INTERPRETATION
    # ------------------------------------------------------------
    story.append(Paragraph("2. RUL Interpretation", heading_style))
    story.append(Paragraph(
        f"<b>Predicted RUL</b> represents the main output of the Random Forest regression "
        f"model and is expressed in operating cycles. For this assessment, the predicted "
        f"value is approximately <b>{rul_cycles} cycles</b>.",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Model Prediction Range</b> ({prediction_range}) represents the variation "
        "between individual decision-tree predictions in the Random Forest. It provides "
        "an indication of model-output variability, but it is <b>not a formal statistical "
        "confidence interval</b> and should not be interpreted as a guaranteed operating-life range.",
        body_style
    ))

    # ------------------------------------------------------------
    # CONDITION ASSESSMENT
    # ------------------------------------------------------------
    story.append(Paragraph("3. Engine Condition Assessment", heading_style))

    condition_explanation = {
        "Healthy": (
            "The predicted RUL is above 50 cycles. Under the project's decision rules, "
            "the engine is considered to have relatively high remaining life and routine "
            "monitoring is appropriate."
        ),
        "Warning": (
            "The predicted RUL is between 20 and 50 cycles. Under the project's decision "
            "rules, the engine has a moderate remaining life and preventive maintenance "
            "planning is recommended."
        ),
        "Critical": (
            "The predicted RUL is below 20 cycles. Under the project's decision rules, "
            "the engine has limited remaining life and immediate inspection and maintenance "
            "planning are recommended."
        )
    }

    story.append(Paragraph(
        f"<b>Classification: {condition}</b><br/>{condition_explanation.get(condition, '')}",
        body_style
    ))

    rules_data = [
        ["Predicted RUL", "Condition", "Risk Level", "Action"],
        ["> 50 cycles", "Healthy", "Low", "Routine monitoring"],
        ["20 – 50 cycles", "Warning", "Medium", "Preventive maintenance"],
        ["< 20 cycles", "Critical", "High", "Immediate inspection"]
    ]
    rules_table = Table(rules_data, colWidths=[100, 105, 95, 140], repeatRows=1)
    rules_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), 8)
    ]))
    story.append(rules_table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "These condition thresholds are project-defined decision rules and are not official NASA labels.",
        small_style
    ))

    # ------------------------------------------------------------
    # CURRENT PARAMETERS + TYPICAL RANGES
    # ------------------------------------------------------------
    story.append(Paragraph("4. Current Engine Parameters", heading_style))
    story.append(Paragraph(
        "The following values were entered into the application for the current prediction. "
        "Typical values correspond to the training-data medians used by the application. "
        "The ranges shown below are the observed ranges in the FD001 training dataset for "
        "the monitored parameters.",
        body_style
    ))

    parameter_rows = [
        ["Parameter", "Current", "Typical", "Observed Range", "Status"],
    ]

    monitored = [
        ("Engine Cycle", cycle, 104.0, 1.0, 362.0),
        ("LPT Outlet Temperature (Sensor 4)", sensor_4, 1408.040, 1382.250, 1441.490),
        ("Physical Core Speed (Sensor 9)", sensor_9, 9060.660, 9021.730, 9244.590),
        ("HPC Outlet Static Pressure (Sensor 11)", sensor_11, 47.510, 46.850, 48.530),
        ("Fuel Flow / Pressure Ratio (Sensor 12)", sensor_12, 521.480, 518.690, 523.380),
        ("Corrected Core Speed (Sensor 14)", sensor_14, 8140.540, 8099.940, 8293.720),
    ]

    for name, current, typical, minimum, maximum in monitored:
        status = "Within observed range" if minimum <= current <= maximum else "Outside observed range"
        parameter_rows.append([
            name,
            f"{current:.3f}" if name != "Engine Cycle" else f"{current:.0f}",
            f"{typical:.3f}" if name != "Engine Cycle" else f"{typical:.0f}",
            f"{minimum:.3f} – {maximum:.3f}" if name != "Engine Cycle" else f"{minimum:.0f} – {maximum:.0f}",
            status
        ])

    parameter_table = Table(parameter_rows, colWidths=[150, 65, 65, 105, 95], repeatRows=1)
    parameter_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5)
    ]))
    story.append(parameter_table)
    story.append(Spacer(1, 12))

    # ------------------------------------------------------------
    # SENSOR INTERPRETATION
    # ------------------------------------------------------------
    story.append(Paragraph("5. Sensor Monitoring Interpretation", heading_style))
    story.append(Paragraph(
        "The monitored sensor readings are compared with their observed FD001 training-data "
        "ranges. Being within the observed range does not by itself prove that the engine is "
        "healthy; the RUL model uses 18 features in total, while only six user-facing values "
        "are entered directly and the remaining model inputs are supplied using training medians.",
        body_style
    ))

    sensor_notes = []
    for name, current, typical, minimum, maximum in monitored[1:]:
        if current < minimum or current > maximum:
            sensor_notes.append(f"{name} is outside the observed training-data range.")
        else:
            sensor_notes.append(f"{name} is within the observed training-data range.")
    story.append(Paragraph("<br/>".join(["<b>Monitoring observations:</b>"] + ["• " + x for x in sensor_notes]), body_style))

    # ------------------------------------------------------------
    # MAINTENANCE RECOMMENDATION
    # ------------------------------------------------------------
    story.append(Paragraph("6. Maintenance Recommendation", heading_style))
    story.append(Paragraph(
        f"<b>Maintenance Priority: {priority}</b><br/>{recommendation}",
        body_style
    ))

    if condition == "Healthy":
        follow_up = (
            "Continue routine condition monitoring and follow the planned maintenance schedule. "
            "Future readings should continue to be monitored for changes in operating behaviour."
        )
    elif condition == "Warning":
        follow_up = (
            "Increase monitoring frequency, review trends in the monitored parameters, and plan "
            "preventive maintenance before the predicted remaining life becomes critical."
        )
    else:
        follow_up = (
            "Prioritize an engineering review and maintenance planning. The prediction should be "
            "used as a decision-support signal rather than as a standalone certification of airworthiness."
        )

    story.append(Paragraph(follow_up, body_style))

    # ------------------------------------------------------------
    # MODEL & METHODOLOGY
    # ------------------------------------------------------------
    story.append(Paragraph("7. Model & Methodology", heading_style))
    story.append(Paragraph(
        "The application uses a Random Forest Regressor trained on the NASA C-MAPSS FD001 "
        "simulated turbofan-engine dataset. The training workflow calculates RUL from engine "
        "cycle history, checks data quality, removes constant features, performs engine-level "
        "train/validation splitting, and trains the regression model using 18 selected features.",
        body_style
    ))

    model_data = [
        ["Model Information", "Details"],
        ["Dataset", "NASA C-MAPSS FD001"],
        ["Training Engines", "100"],
        ["Training Records", "20,631"],
        ["Operating Conditions", "1"],
        ["Fault Mode", "1"],
        ["Algorithm", "Random Forest Regressor"],
        ["Model Features", "18"],
        ["Decision Trees", "50"],
        ["Validation MAE", "23.78 cycles"],
        ["Validation RMSE", "31.31 cycles"],
        ["Validation R²", "0.7725"]
    ]
    model_table = Table(model_data, colWidths=[180, 270], repeatRows=1)
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#70AD47")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5)
    ]))
    story.append(model_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "RUL calculation in the training data: <b>RUL = Maximum Cycle of Engine − Current Cycle</b>.",
        body_style
    ))

    # ------------------------------------------------------------
    # LIMITATIONS / DISCLAIMER
    # ------------------------------------------------------------
    story.append(Paragraph("8. Limitations and Disclaimer", heading_style))
    story.append(Paragraph(
        "The NASA C-MAPSS FD001 dataset is a simulated turbofan-engine dataset. The sensor "
        "measurements and operating conditions are dataset-specific. The application is a "
        "machine-learning project demonstration and is not a certified aviation maintenance "
        "or airworthiness system. The prediction should not be treated as a guarantee of the "
        "number of cycles an engine can safely operate. Actual maintenance decisions require "
        "appropriate engineering procedures, inspection results, operational records, and "
        "qualified aviation personnel.",
        body_style
    ))

    story.append(Paragraph(
        "Generated by the Aircraft Engine Remaining Useful Life Prediction and Condition Monitoring system.",
        small_style
    ))

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

        "symbol": "φ",

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

    st.subheader("📊 Dataset")

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

    st.subheader("⚙️ System")

    st.write("✓ RUL Prediction")

    st.write("✓ Risk Assessment")

    st.write("✓ Sensor Monitoring")

    st.write("✓ Dataset Explorer")

    st.write("✓ Maintenance Recommendation")


# ============================================================
# MAIN NAVIGATION
# ============================================================

st.title(
    "✈️ Aircraft Engine Predictive Maintenance"
)

st.caption(
    "Remaining Useful Life Prediction & Condition Monitoring"
)

page = st.radio(
    "Navigation",
    [
        "🏠 Prediction Dashboard",
        "📂 Dataset Explorer",
        "🤖 Model Information"
    ],
    horizontal=True
)


# ============================================================
# ============================================================
# PAGE 1 — PREDICTION DASHBOARD
# ============================================================
# ============================================================

if page == "🏠 Prediction Dashboard":

    st.subheader(
        "🔧 Current Engine Parameters"
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

            min_value=1.0,

            max_value=362.0,

            value=104.0,

            step=1.0

        )

        st.caption(
            "Range: 1 – 362 | Typical: 104"
        )


    # --------------------------------------------------------
    # SENSOR 11
    # --------------------------------------------------------

    with col2:

        sensor_11 = st.number_input(

            "HPC Outlet Static Pressure (Sensor 11)",

            min_value=46.850,

            max_value=48.530,

            value=47.510,

            step=0.001

        )

        st.caption(
            "Range: 46.850 – 48.530 | Typical: 47.510"
        )


    # --------------------------------------------------------
    # SENSOR 9
    # --------------------------------------------------------

    with col3:

        sensor_9 = st.number_input(

            "Physical Core Speed (Sensor 9)",

            min_value=9021.730,

            max_value=9244.590,

            value=9060.660,

            step=0.001

        )

        st.caption(
            "Range: 9021.730 – 9244.590 | Typical: 9060.660"
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

            min_value=1382.250,

            max_value=1441.490,

            value=1408.040,

            step=0.001

        )

        st.caption(
            "Range: 1382.250 – 1441.490 | Typical: 1408.040"
        )


    # --------------------------------------------------------
    # SENSOR 14
    # --------------------------------------------------------

    with col5:

        sensor_14 = st.number_input(

            "Corrected Core Speed (Sensor 14)",

            min_value=8099.940,

            max_value=8293.720,

            value=8140.540,

            step=0.001

        )

        st.caption(
            "Range: 8099.940 – 8293.720 | Typical: 8140.540"
        )


    # --------------------------------------------------------
    # SENSOR 12
    # --------------------------------------------------------

    with col6:

        sensor_12 = st.number_input(

            "Fuel Flow / Pressure Ratio (Sensor 12)",

            min_value=518.690,

            max_value=523.380,

            value=521.480,

            step=0.001

        )

        st.caption(
            "Range: 518.690 – 523.380 | Typical: 521.480"
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown("---")

    predict_button = st.button(

        "🚀 Analyze Engine Condition",

        use_container_width=True

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:


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

            f"{round(lower_bound)} – "
            f"{round(upper_bound)} cycles"

        )


        # ====================================================
        # ENGINE CONDITION
        # ====================================================

        if predicted_rul > 50:

            condition = "Healthy"

            risk = "Low"

            priority = "Routine"

            icon = "🟢"

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

            icon = "🟡"

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

                sensor_icon = "🟢"


            elif deviation < 40:

                sensor_status = "Moderate Deviation"

                sensor_icon = "🟡"


            else:

                sensor_status = "High Deviation"

                sensor_icon = "🔴"


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
            "📊 Engine Monitoring Dashboard"
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

                "✈️ Predicted RUL",

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


        # ====================================================
        # RUL VISUAL
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🔋 Remaining Engine Life"
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
            "🎯 Model Prediction Range"
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
            "🔬 Sensor Condition Monitoring"
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
            "🛠️ Maintenance Recommendation"
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
            "📋 Current Engine Parameters"
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
            "📝 AI Monitoring Summary"
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
            "📥 Download PDF Report",
            data=pdf_data,
            file_name="aircraft_engine_rul_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# ============================================================
# PAGE 2 — DATASET EXPLORER
# ============================================================
# ============================================================

elif page == "📂 Dataset Explorer":

    st.header(
        "📂 NASA C-MAPSS FD001 Dataset Explorer"
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
            "🔎 Dataset Quality"
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
            "📄 Dataset Preview"
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
            "🔍 Explore Individual Engine"
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
            "🧮 How RUL Is Calculated"
        )

        st.code(
            "RUL = Maximum Cycle of Engine - Current Cycle"
        )

        st.write(
            "For example, if an engine's maximum training "
            "cycle is 200 and its current cycle is 150:"
        )

        st.info(
            "RUL = 200 − 150 = 50 cycles"
        )


        # ====================================================
        # COLUMN INFORMATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📋 Dataset Columns"
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
# PAGE 3 — MODEL INFORMATION
# ============================================================
# ============================================================

else:

    st.header(
        "🤖 Machine Learning Model Information"
    )

    st.write(
        "Technical overview of the preprocessing, "
        "model and evaluation used in the project."
    )


    # ========================================================
    # MODEL OVERVIEW
    # ========================================================

    st.subheader(
        "🌳 Model Overview"
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
        "📊 Dataset"
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
        "⚙️ Data Preprocessing"
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
            f"✓ {step}"
        )


    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔢 Model Features"
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
        "📈 Model Performance"
    )


    performance_df = pd.DataFrame({

        "Metric": [

            "Validation MAE",

            "Validation RMSE",

            "Validation R²"

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
        "🚦 Engine Condition Rules"
    )


    condition_df = pd.DataFrame({

        "Predicted RUL": [

            "> 50 cycles",

            "20 – 50 cycles",

            "< 20 cycles"

        ],

        "Condition": [

            "🟢 Healthy",

            "🟡 Warning",

            "🔴 Critical"

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
        "🔄 Project Workflow"
    )


    st.write(
        "NASA C-MAPSS Dataset"
    )

    st.write(
        "↓"
    )

    st.write(
        "Data Preprocessing"
    )

    st.write(
        "↓"
    )

    st.write(
        "Feature Selection"
    )

    st.write(
        "↓"
    )

    st.write(
        "Random Forest Regression"
    )

    st.write(
        "↓"
    )

    st.write(
        "RUL Prediction"
    )

    st.write(
        "↓"
    )

    st.write(
        "Risk & Condition Assessment"
    )

    st.write(
        "↓"
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
