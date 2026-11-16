from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

RAW_DATA_PATH = DATA_DIR / "student_mental_health.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_ds.csv"
MODEL_PATH = MODEL_DIR / "mental_health_LR_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"




TARGET_COL = "Mental_Health_Risk"
DROP_COLS = ["Student_ID", "Age"]




NUMERICAL_COLS = [
    "Age",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
    "Year_of_Study"
]

ONE_HOT_COLS = ["Gender"]

ORDINAL_COLS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    "Part_Time_Job": ["No", "Yes"],
    "Previous_Mental_Health_History": ["No", "Yes"]
}

RANDOM_STATE = 42
TEST_SIZE = 0.2