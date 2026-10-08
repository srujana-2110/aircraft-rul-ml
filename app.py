import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Aircraft Engine RUL Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_FILE = "aircraft_rul_random_forest.pkl"
FEATURE_FILE = "rul_features.pkl"
TRAIN_FILE = "train_FD001.txt"

DATA_COLUMNS = [
    "unit_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

MODEL_FEATURES = [
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

DEFAULT_VALUES = {
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

INPUT_RANGES = {
    "cycle": (1.0, 362.0),
    "sensor_11": (46.850, 48.530),
    "sensor_9": (9021.730, 9244.590),
    "sensor_4": (1382.250, 1441.490),
    "sensor_14": (8099.940, 8293.720),
    "sensor_12": (518.690, 523.380)
}

INPUT_LABELS = {
    "cycle": "Engine Cycle",
    "sensor_11": "Sensor 11",
    "sensor_9": "Sensor 9",
    "sensor_4": "Sensor 4",
    "sensor_14": "Sensor 14",
    "sensor_12": "Sensor 12"
}

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main-title {
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #666;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}
.result-card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #ddd;
    background: #fafafa;
    margin-bottom: 15px;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 650;
    margin-top: 10px;
}
.small-note {
    color: #666;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_resource
def load_features():
    return joblib.load(FEATURE_FILE)

try:
    model = load_model()
    saved_features = load_features()
except Exception as e:
    st.error(
        "Unable to load the trained model files. Make sure "
        "'aircraft_rul_random_forest.pkl' and 'rul_features.pkl' "
        "are in the same folder as app.py."
    )
    st.exception(e)
    st.stop()

# ============================================================
# LOAD TRAINING DATA
# ============================================================
@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv(
            TRAIN_FILE,
            sep=r"\s+",
            header=None
        )
        df.columns = DATA_COLUMNS
        return df
    except Exception:
        return None

train_df = load_training_data()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def classify_condition(rul):
    if rul > 50:
        return "Healthy", "Low", "Routine"
    elif rul >= 20:
        return "Warning", "Medium", "Preventive"
    else:
        return "Critical", "High", "Immediate"

def get_recommendation(condition):
    if condition == "Healthy":
        return (
            "Continue routine monitoring and scheduled maintenance. "
            "No immediate maintenance action is required."
        )
    elif condition == "Warning":
        return (
            "Increase monitoring frequency and schedule preventive "
            "maintenance to reduce the risk of unexpected failure."
        )
    else:
        return (
            "Immediate inspection is recommended. Prioritize "
            "maintenance planning and investigate the engine condition."
        )

def build_input_dataframe(user_values):
    values = DEFAULT_VALUES.copy()

    for key, value in user_values.items():
        values[key] = value

    # Ensure exact feature order used during training.
    input_df = pd.DataFrame(
        [[values[feature] for feature in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )
    return input_df

def predict_engine(user_values):
    input_df = build_input_dataframe(user_values)

    predicted_rul = float(model.predict(input_df)[0])
    predicted_rul = max(0.0, predicted_rul)

    # Variation across individual Random Forest trees.
    tree_predictions = np.array([
        float(tree.predict(input_df)[0])
        for tree in model.estimators_
    ])

    lower = max(0.0, float(np.percentile(tree_predictions, 10)))
    upper = max(0.0, float(np.percentile(tree_predictions, 90)))

    condition, risk, priority = classify_condition(predicted_rul)
    recommendation = get_recommendation(condition)

    return {
        "predicted_rul": predicted_rul,
        "lower": lower,
        "upper": upper,
        "condition": condition,
        "risk": risk,
        "priority": priority,
        "recommendation": recommendation,
        "input_df": input_df
    }

def create_pdf_report(result, user_values):
    """Generate a professional PDF maintenance analysis report."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14
    )

    story = []

    story.append(Paragraph(
        "Aircraft Engine Remaining Useful Life (RUL) Analysis Report",
        title_style
    ))
    story.append(Paragraph(
        "Condition Monitoring and Predictive Maintenance Report",
        subtitle_style
    ))

    # Executive summary
    story.append(Paragraph("1. Executive Summary", heading_style))

    summary_data = [
        ["Metric", "Result"],
        ["Predicted RUL", f"{result['predicted_rul']:.1f} cycles"],
        ["Model Prediction Range",
         f"{result['lower']:.1f} – {result['upper']:.1f} cycles"],
        ["Engine Condition", result["condition"]],
        ["Risk Level", result["risk"]],
        ["Maintenance Priority", result["priority"]],
    ]

    summary_table = Table(summary_data, colWidths=[70 * mm, 90 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # Assessment
    story.append(Paragraph("2. Condition Assessment", heading_style))
    story.append(Paragraph(
        result["recommendation"],
        body_style
    ))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Condition thresholds used in this project: "
        "Healthy &gt; 50 cycles, Warning 20–50 cycles, "
        "and Critical &lt; 20 cycles.",
        body_style
    ))

    # Input parameters
    story.append(Paragraph("3. Current Engine Parameters", heading_style))

    parameter_rows = [["Parameter", "Current Value", "Training Range"]]

    for key in [
        "cycle", "sensor_11", "sensor_9",
        "sensor_4", "sensor_14", "sensor_12"
    ]:
        low, high = INPUT_RANGES[key]
        value = user_values[key]

        if key == "cycle":
            value_text = f"{value:.0f}"
            range_text = f"{low:.0f} – {high:.0f}"
        else:
            value_text = f"{value:.3f}"
            range_text = f"{low:.3f} – {high:.3f}"

        parameter_rows.append([
            INPUT_LABELS[key],
            value_text,
            range_text
        ])

    parameter_table = Table(
        parameter_rows,
        colWidths=[65 * mm, 45 * mm, 55 * mm]
    )

    parameter_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(parameter_table)

    # Model details
    story.append(Paragraph("4. Machine Learning Model", heading_style))

    model_data = [
        ["Model", "Random Forest Regressor"],
        ["Deployed Trees", str(len(model.estimators_))],
        ["Model Features", str(len(MODEL_FEATURES))],
        ["Dataset", "NASA C-MAPSS FD001"],
        ["Prediction Target", "Remaining Useful Life (cycles)"],
        ["Application", "Aircraft Engine Condition Monitoring"],
    ]

    model_table = Table(model_data, colWidths=[60 * mm, 105 * mm])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2f8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(model_table)

    story.append(Paragraph("5. Prediction Method", heading_style))
    story.append(Paragraph(
        "The application accepts six user-facing engine parameters. "
        "The trained model requires 18 features, so the remaining model "
        "features are populated using their training-data median values. "
        "The resulting feature vector is passed to the trained Random "
        "Forest Regressor to estimate remaining useful life.",
        body_style
    ))

    story.append(Spacer(1, 6))

    story.append(Paragraph("6. Model Prediction Range", heading_style))
    story.append(Paragraph(
        f"The predicted RUL is {result['predicted_rul']:.1f} cycles. "
        f"The 10th–90th percentile range of individual Random Forest "
        f"tree predictions is {result['lower']:.1f}–"
        f"{result['upper']:.1f} cycles. This range represents variation "
        f"among the model's individual trees and should not be interpreted "
        f"as a formal statistical confidence interval.",
        body_style
    ))

    story.append(Spacer(1, 6))

    story.append(Paragraph("7. Maintenance Recommendation", heading_style))
    story.append(Paragraph(
        result["recommendation"],
        body_style
    ))

    # Disclaimer
    story.append(Spacer(1, 14))
    story.append(Paragraph("8. Report Notes", heading_style))
    story.append(Paragraph(
        "This report is generated by a machine-learning demonstration "
        "application using the NASA C-MAPSS FD001 simulated dataset. "
        "The sensor columns are anonymized dataset indicators. The "
        "prediction is intended for project-level condition monitoring "
        "and should not be treated as a certified aviation maintenance "
        "decision or a guaranteed failure forecast.",
        body_style
    ))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(
            A4[0] / 2,
            8 * mm,
            f"Aircraft Engine RUL Analysis Report  |  Page {doc.page}"
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("✈️ Aircraft RUL")
st.sidebar.caption("Predictive Maintenance System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Prediction Dashboard",
        "📂 Dataset Explorer",
        "🤖 Model Information"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "NASA C-MAPSS FD001 is a simulated aircraft engine "
    "degradation dataset used for RUL prediction."
)

# ============================================================
# PAGE 1 — PREDICTION DASHBOARD
# ============================================================
if page == "🏠 Prediction Dashboard":

    st.markdown(
        '<div class="main-title">✈️ Aircraft Engine RUL Prediction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Machine-learning based condition monitoring and predictive maintenance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🔧 Enter Engine Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        cycle = st.number_input(
            "Engine Cycle",
            min_value=1.0,
            max_value=362.0,
            value=104.0,
            step=1.0,
            help="Current operating cycle of the engine. Range: 1–362."
        )
        st.caption("Range: 1 – 362")

        sensor_11 = st.number_input(
            "Sensor 11",
            min_value=46.850,
            max_value=48.530,
            value=47.510,
            step=0.001,
            format="%.3f"
        )
        st.caption("Range: 46.850 – 48.530")

    with col2:
        sensor_9 = st.number_input(
            "Sensor 9",
            min_value=9021.730,
            max_value=9244.590,
            value=9060.660,
            step=0.001,
            format="%.3f"
        )
        st.caption("Range: 9021.730 – 9244.590")

        sensor_4 = st.number_input(
            "Sensor 4",
            min_value=1382.250,
            max_value=1441.490,
            value=1408.040,
            step=0.001,
            format="%.3f"
        )
        st.caption("Range: 1382.250 – 1441.490")

    with col3:
        sensor_14 = st.number_input(
            "Sensor 14",
            min_value=8099.940,
            max_value=8293.720,
            value=8140.540,
            step=0.001,
            format="%.3f"
        )
        st.caption("Range: 8099.940 – 8293.720")

        sensor_12 = st.number_input(
            "Sensor 12",
            min_value=518.690,
            max_value=523.380,
            value=521.480,
            step=0.001,
            format="%.3f"
        )
        st.caption("Range: 518.690 – 523.380")

    st.markdown("---")

    predict_clicked = st.button(
        "🔮 Predict Engine RUL",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:
        user_values = {
            "cycle": cycle,
            "sensor_11": sensor_11,
            "sensor_9": sensor_9,
            "sensor_4": sensor_4,
            "sensor_14": sensor_14,
            "sensor_12": sensor_12
        }

        result = predict_engine(user_values)

        # Save latest result so report remains available on reruns.
        st.session_state["latest_result"] = result
        st.session_state["latest_user_values"] = user_values

    if "latest_result" in st.session_state:

        result = st.session_state["latest_result"]
        user_values = st.session_state["latest_user_values"]

        st.markdown("## 📊 Prediction Result")

        # Main result cards
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Predicted RUL",
                f"{result['predicted_rul']:.1f} cycles"
            )

        with c2:
            st.metric(
                "Engine Condition",
                result["condition"]
            )

        with c3:
            st.metric(
                "Risk Level",
                result["risk"]
            )

        with c4:
            st.metric(
                "Maintenance Priority",
                result["priority"]
            )

        # RUL progress
        st.markdown("### 📈 RUL Monitoring")

        # Display relative to a 0–100+ cycle monitoring scale.
        progress_value = int(
            min(100, max(0, result["predicted_rul"]))
        )
        st.progress(progress_value)

        st.caption(
            f"Predicted remaining useful life: "
            f"{result['predicted_rul']:.1f} cycles"
        )

        # Prediction range
        range_col1, range_col2 = st.columns(2)

        with range_col1:
            st.info(
                f"**Model Prediction Range**\n\n"
                f"{result['lower']:.1f} – {result['upper']:.1f} cycles"
            )

        with range_col2:
            st.info(
                "**Interpretation**\n\n"
                "The range represents variation among individual "
                "Random Forest tree predictions."
            )

        # Condition message
        if result["condition"] == "Healthy":
            st.success(
                f"🟢 **Healthy:** {result['recommendation']}"
            )
        elif result["condition"] == "Warning":
            st.warning(
                f"🟠 **Warning:** {result['recommendation']}"
            )
        else:
            st.error(
                f"🔴 **Critical:** {result['recommendation']}"
            )

        # Current parameters
        st.markdown("### 🔧 Current Engine Parameters")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric("Engine Cycle", f"{cycle:.0f}")
            st.metric("Sensor 11", f"{sensor_11:.3f}")

        with p2:
            st.metric("Sensor 9", f"{sensor_9:.3f}")
            st.metric("Sensor 4", f"{sensor_4:.3f}")

        with p3:
            st.metric("Sensor 14", f"{sensor_14:.3f}")
            st.metric("Sensor 12", f"{sensor_12:.3f}")

        # Monitoring table
        st.markdown("### 🔍 Sensor Condition Monitoring")

        monitoring_rows = []

        for key in [
            "cycle", "sensor_11", "sensor_9",
            "sensor_4", "sensor_14", "sensor_12"
        ]:
            low, high = INPUT_RANGES[key]
            current = user_values[key]
            median = DEFAULT_VALUES[key]

            if low <= current <= high:
                status = "Within training range"
            else:
                status = "Outside training range"

            monitoring_rows.append({
                "Parameter": INPUT_LABELS[key],
                "Current Value": (
                    f"{current:.0f}"
                    if key == "cycle"
                    else f"{current:.3f}"
                ),
                "Training Median": (
                    f"{median:.0f}"
                    if key == "cycle"
                    else f"{median:.3f}"
                ),
                "Training Range": (
                    f"{low:.0f} – {high:.0f}"
                    if key == "cycle"
                    else f"{low:.3f} – {high:.3f}"
                ),
                "Status": status
            })

        st.dataframe(
            pd.DataFrame(monitoring_rows),
            use_container_width=True,
            hide_index=True
        )

        # Maintenance recommendation
        st.markdown("### 🛠️ Maintenance Recommendation")

        st.write(result["recommendation"])

        # PDF report
        st.markdown("---")
        st.markdown("## 📄 Generate Maintenance Analysis Report")

        st.write(
            "Download the current prediction, engine parameters, "
            "model information, condition assessment, and maintenance "
            "recommendation as a professional PDF report."
        )

        pdf_bytes = create_pdf_report(result, user_values)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="Aircraft_Engine_RUL_Analysis_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.caption(
            "The PDF is generated dynamically from the latest prediction."
        )

# ============================================================
# PAGE 2 — DATASET EXPLORER
# ============================================================
elif page == "📂 Dataset Explorer":

    st.title("📂 Dataset Explorer")

    if train_df is None:
        st.error(
            "train_FD001.txt was not found. Add the dataset file "
            "to the same folder as app.py to enable this section."
        )
        st.stop()

    # Calculate RUL
    explorer_df = train_df.copy()
    max_cycles = explorer_df.groupby("unit_id")["cycle"].transform("max")
    explorer_df["RUL"] = max_cycles - explorer_df["cycle"]

    st.markdown("### 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Training Records", f"{len(explorer_df):,}")

    with c2:
        st.metric("Engines", f"{explorer_df['unit_id'].nunique():,}")

    with c3:
        st.metric("Features", f"{len(DATA_COLUMNS) - 1}")

    with c4:
        st.metric("Maximum RUL", f"{explorer_df['RUL'].max():.0f} cycles")

    st.markdown("### 🧹 Data Quality")

    q1, q2 = st.columns(2)

    with q1:
        st.metric(
            "Missing Values",
            f"{int(explorer_df.isna().sum().sum()):,}"
        )

    with q2:
        st.metric(
            "Duplicate Rows",
            f"{int(explorer_df.duplicated().sum()):,}"
        )

    st.markdown("### 👀 Dataset Preview")

    st.dataframe(
        explorer_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.markdown("### 🔍 Engine History Explorer")

    selected_engine = st.selectbox(
        "Select Engine ID",
        sorted(explorer_df["unit_id"].unique())
    )

    engine_history = explorer_df[
        explorer_df["unit_id"] == selected_engine
    ].copy()

    st.write(
        f"Engine **{selected_engine}** contains "
        f"**{len(engine_history)} cycles**."
    )

    st.dataframe(
        engine_history,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🧮 RUL Calculation")

    st.code(
        "RUL = Maximum Cycle of Engine - Current Cycle",
        language="text"
    )

    st.write(
        "For each engine, the maximum observed cycle is treated as "
        "the final cycle in the training sequence. RUL is calculated "
        "as the difference between that maximum cycle and the current cycle."
    )

    st.markdown("### 📌 Model Features")

    st.dataframe(
        pd.DataFrame({
            "Model Feature": MODEL_FEATURES
        }),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# PAGE 3 — MODEL INFORMATION
# ============================================================
else:

    st.title("🤖 Model Information")

    st.markdown("### 🧠 Model Overview")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric("Algorithm", "Random Forest")

    with info2:
        st.metric("Trees", str(len(model.estimators_)))

    with info3:
        st.metric("Model Features", str(len(MODEL_FEATURES)))

    st.markdown("### 📚 Dataset Information")

    st.write("""
    **NASA C-MAPSS FD001** is a simulated aircraft engine degradation
    dataset designed for prognostics and Remaining Useful Life prediction.
    FD001 contains one operating condition and one fault mode.
    """)

    st.markdown("### 🔄 Preprocessing Workflow")

    st.markdown("""
    1. Load the NASA C-MAPSS FD001 training data.
    2. Assign column names to the raw dataset.
    3. Calculate training RUL using maximum cycle minus current cycle.
    4. Check missing values and duplicate records.
    5. Remove constant features.
    6. Keep 18 useful model features.
    7. Split data at the engine level to reduce data leakage.
    8. Train the Random Forest regression model.
    9. Save the trained model using Joblib.
    10. Deploy the model through Streamlit.
    """)

    st.markdown("### 📈 Validation Performance")

    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": ["23.78 cycles", "31.31 cycles", "0.7725"]
    })

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🚦 Condition Rules")

    rules_df = pd.DataFrame({
        "Predicted RUL": ["> 50 cycles", "20 – 50 cycles", "< 20 cycles"],
        "Condition": ["Healthy", "Warning", "Critical"],
        "Risk": ["Low", "Medium", "High"],
        "Maintenance Priority": ["Routine", "Preventive", "Immediate"]
    })

    st.dataframe(
        rules_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🏗️ Application Workflow")

    st.code(
        """User Input
    ↓
18-Feature Model Input
    ↓
Random Forest Regression
    ↓
Predicted RUL
    ↓
Condition Classification
    ↓
Risk & Maintenance Priority
    ↓
Condition Monitoring Dashboard
    ↓
PDF Maintenance Report""",
        language="text"
    )

    st.markdown("### ⚠️ Project Note")

    st.info(
        "The sensor columns in C-MAPSS are anonymized dataset indicators. "
        "This application is intended for project-level predictive "
        "maintenance demonstration and not as a certified aviation "
        "maintenance decision system."
    )
