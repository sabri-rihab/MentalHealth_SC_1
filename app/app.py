"""
Streamlit Web Application: Student Mental Health Risk Predictor
Provides an interactive, simple user interface for real-time risk predictions.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Mental Health Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------------------------------------------------
# Paths & Model Loading
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "mental_health_LR_model.joblib"


@st.cache_resource
def load_trained_model(path: Path):
    """Loads the trained model pipeline from disk."""
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_trained_model(MODEL_PATH)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🧠 Student Mental Health Risk Predictor")
st.markdown(
    """
    This tool uses a trained machine learning model to assess student mental health risk levels 
    (**Low**, **Medium**, or **High**) based on academic, lifestyle, and psychosocial indicators.
    """
)
st.write("---")

if model is None:
    st.warning(
        f"⚠️ **Model file not found at:** `{MODEL_PATH}`\n\n"
        "Please train the model first by running the Airflow training DAG or executing `python src/train.py`."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------
st.subheader("📋 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🎓 Academics & Demographics")
    gender = st.selectbox("Gender", options=["Male", "Female", "Non-binary"])
    year_of_study = st.selectbox("Year of Study", options=[1, 2, 3, 4, 5], index=1)
    gpa = st.number_input(
        "Cumulative GPA (0.0 - 4.0)",
        min_value=0.0,
        max_value=4.0,
        value=3.20,
        step=0.05
    )
    study_hours = st.number_input(
        "Study Hours / Week",
        min_value=0.0,
        max_value=80.0,
        value=20.0,
        step=1.0
    )
    attendance_rate = st.slider(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

with col2:
    st.markdown("##### 🌿 Lifestyle & Habits")
    sleep_hours = st.number_input(
        "Nightly Sleep (Hours)",
        min_value=0.0,
        max_value=16.0,
        value=7.0,
        step=0.5
    )
    social_media_hours = st.number_input(
        "Social Media Hours / Day",
        min_value=0.0,
        max_value=16.0,
        value=3.0,
        step=0.5
    )
    screen_time_hours = st.number_input(
        "Total Screen Time / Day (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.5
    )
    physical_activity = st.selectbox(
        "Physical Activity Level",
        options=["Low", "Moderate", "High"],
        index=1
    )
    diet_quality = st.selectbox(
        "Diet Quality",
        options=["Poor", "Average", "Good"],
        index=1
    )

st.write("")
st.markdown("##### 🤝 Psychosocial & Personal Factors")
col3, col4 = st.columns(2)

with col3:
    financial_stress = st.slider(
        "Financial Stress Level (1 = Low, 10 = High)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    family_support = st.slider(
        "Perceived Family Support (1 = Low, 10 = High)",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )

with col4:
    part_time_job = st.selectbox("Has Part-Time Job?", options=["No", "Yes"])
    previous_history = st.selectbox(
        "Previous Mental Health Condition History?",
        options=["No", "Yes"]
    )

# Sanity adjustment: Screen Time cannot be less than Social Media Hours
if screen_time_hours < social_media_hours:
    st.info(
        "💡 *Note:* Total screen time was less than social media time. "
        "It will be adjusted to equal social media hours during prediction."
    )
    screen_time_hours = social_media_hours

# -----------------------------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------------------------
def predict_risk(trained_model, input_data: pd.DataFrame):
    """
    Executes prediction handling both:
    1. Full Pipeline (with ColumnTransformer preprocessor)
    2. Baseline Pipeline (expects pre-encoded feature columns)
    """
    # Case 1: Full Pipeline (Preferred)
    if hasattr(trained_model, "named_steps") and "preprocessor" in trained_model.named_steps:
        pred = trained_model.predict(input_data)[0]
        proba = trained_model.predict_proba(input_data)[0] if hasattr(trained_model, "predict_proba") else None
        return pred, proba, getattr(trained_model, "classes_", None)

    # Case 2: Baseline Pipeline (manual encoding fallback)
    encoded = input_data.copy()
    
    # Ordinal mappings
    ordinal_mapping = {
        "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
        "Diet_Quality": {"Poor": 0, "Average": 1, "Good": 2},
        "Part_Time_Job": {"No": 0, "Yes": 1},
        "Previous_Mental_Health_History": {"No": 0, "Yes": 1}
    }
    for col, mapping in ordinal_mapping.items():
        if col in encoded.columns:
            encoded[col] = encoded[col].map(mapping)

    # One-hot encoding for Gender
    encoded["Gender_Male"] = 1 if input_data["Gender"].iloc[0] == "Male" else 0
    encoded["Gender_Non-binary"] = 1 if input_data["Gender"].iloc[0] == "Non-binary" else 0
    encoded = encoded.drop(columns=["Gender", "Student_ID", "Age"], errors="ignore")

    pred = trained_model.predict(encoded)[0]
    proba = trained_model.predict_proba(encoded)[0] if hasattr(trained_model, "predict_proba") else None
    return pred, proba, getattr(trained_model, "classes_", None)


# -----------------------------------------------------------------------------
# Trigger Prediction
# -----------------------------------------------------------------------------
st.write("---")

if st.button("🔮 Predict Mental Health Risk", type="primary", use_container_width=True):
    # Prepare input DataFrame with exact raw feature names
    input_row = pd.DataFrame([{
        "Gender": gender,
        "Year_of_Study": int(year_of_study),
        "Sleep_Hours": float(sleep_hours),
        "Study_Hours_Per_Week": float(study_hours),
        "GPA": float(gpa),
        "Social_Media_Hours": float(social_media_hours),
        "Screen_Time_Hours": float(screen_time_hours),
        "Physical_Activity": physical_activity,
        "Financial_Stress": float(financial_stress),
        "Family_Support": float(family_support),
        "Attendance_Rate": float(attendance_rate),
        "Part_Time_Job": part_time_job,
        "Diet_Quality": diet_quality,
        "Previous_Mental_Health_History": previous_history
    }])

    prediction, probabilities, classes = predict_risk(model, input_row)

    # Standardize label mapping: 0 -> High, 1 -> Low, 2 -> Medium (or string)
    label_map = {0: "High", 1: "Low", 2: "Medium", "0": "High", "1": "Low", "2": "Medium"}
    predicted_label = label_map.get(prediction, str(prediction))

    st.write("### 🎯 Prediction Result")

    if predicted_label == "Low":
        st.success("### 🟢 Low Mental Health Risk")
        st.write(
            "The student displays positive protective factors: healthy sleep, active routine, "
            "and manageable stress indicators. Continue maintaining current academic-life balance."
        )
    elif predicted_label == "Medium":
        st.warning("### 🟡 Medium Mental Health Risk")
        st.write(
            "The student shows moderate vulnerability markers (e.g., elevated stress, irregular sleep, "
            "or study pressure). Recommended to monitor stress levels and leverage campus support resources."
        )
    else:
        st.error("### 🔴 High Mental Health Risk")
        st.write(
            "The student indicates multiple high-risk factors (e.g., severe sleep disruption, high financial strain, "
            "or prior history). Proactive outreach and consultation with university counseling services are strongly advised."
        )

    # Probability Breakdown
    if probabilities is not None and classes is not None:
        st.write("#### 📊 Confidence Probabilities")
        prob_cols = st.columns(len(classes))
        for idx, cls in enumerate(classes):
            cls_name = label_map.get(cls, str(cls))
            prob_percent = probabilities[idx] * 100
            with prob_cols[idx]:
                st.metric(label=f"{cls_name} Risk", value=f"{prob_percent:.1f}%")

st.write("---")
st.caption(
    "⚠️ **Disclaimer:** This tool is an academic machine learning prototype designed for exploratory "
    "and educational purposes. It does not constitute medical diagnosis or clinical psychological advice."
)

