import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="BP Prediction Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': 'User',
        'age': None,
        'height': None,
        'weight': None,
        'sex': None,
        'medical_conditions': []
    }

if 'database_initialized' not in st.session_state:
    st.session_state.database_initialized = False

# Initialize database
if not st.session_state.database_initialized:
    try:
        from utils.database import initialize_database
        initialize_database()
        st.session_state.database_initialized = True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")

# Main page content
st.title("Blood Pressure Prediction Application")

st.divider()

# Get data from database
try:
    from utils.database import (
        get_user_profile,
        get_prediction_count,
        get_average_bp,
        get_statistics_summary,
        get_all_predictions
    )

    user = get_user_profile()
    stats = get_statistics_summary()
    all_predictions = get_all_predictions()

    has_calibration = False
    avg_sbp_cal = None
    avg_dbp_cal = None

    if all_predictions:
        calibrated_predictions = [p for p in all_predictions if p.get('calibrated_sbp') is not None and p.get('calibrated_dbp') is not None]

        if calibrated_predictions:
            has_calibration = True
            avg_sbp_cal = sum(p['calibrated_sbp'] for p in calibrated_predictions) / len(calibrated_predictions)
            avg_dbp_cal = sum(p['calibrated_dbp'] for p in calibrated_predictions) / len(calibrated_predictions)

    # Calculate BMI
    bmi = None
    bmi_category = None
    if user and user.get('height_cm') and user.get('weight_kg'):
        height_m = user['height_cm'] / 100
        bmi = user['weight_kg'] / (height_m ** 2)
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

    dob = None
    if user and user.get('age'):
        current_year = datetime.now().year
        birth_year = current_year - user['age']
        dob = f"~{birth_year}"

    # Calculate resting heart rate (average all mean_hr values)
    resting_hr = None
    if all_predictions:
        hr_values = [p['mean_hr'] for p in all_predictions if p.get('mean_hr')]
        if hr_values:
            resting_hr = sum(hr_values) / len(hr_values)

    st.subheader("Patient Demographics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Name", user.get('name', 'Not set') if user else 'Not set')
        st.metric("Sex", user.get('sex', 'Not set') if user else 'Not set')

    with col2:
        st.metric("Date of Birth", dob if dob else 'Not set')
        st.metric("Age", f"{user.get('age')} years" if user and user.get('age') else 'Not set')

    with col3:
        st.metric("Height", f"{user.get('height_cm'):.1f} cm" if user and user.get('height_cm') else 'Not set')
        st.metric("Weight", f"{user.get('weight_kg'):.1f} kg" if user and user.get('weight_kg') else 'Not set')

    with col4:
        st.metric("BMI", f"{bmi:.1f}" if bmi else 'Not set')
        st.metric("BMI Category", bmi_category if bmi_category else 'Not set')

    st.divider()

    st.subheader("Cardiovascular Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if has_calibration and avg_sbp_cal:
            st.metric(
                "Average Systolic BP",
                f"{avg_sbp_cal:.0f} mmHg",
                help="Calibrated value"
            )
        elif stats and stats.get('avg_sbp'):
            st.metric(
                "Average Systolic BP",
                f"{stats['avg_sbp']:.0f} mmHg",
                help="Raw prediction (not calibrated)"
            )
        else:
            st.metric("Average Systolic BP", "No data")

    with col2:
        if has_calibration and avg_dbp_cal:
            st.metric(
                "Average Diastolic BP",
                f"{avg_dbp_cal:.0f} mmHg",
                help="Calibrated value"
            )
        elif stats and stats.get('avg_dbp'):
            st.metric(
                "Average Diastolic BP",
                f"{stats['avg_dbp']:.0f} mmHg",
                help="Raw prediction (not calibrated)"
            )
        else:
            st.metric("Average Diastolic BP", "No data")

    with col3:
        if resting_hr:
            st.metric("Resting Heart Rate", f"{resting_hr:.0f} BPM")
        else:
            st.metric("Resting Heart Rate", "No data")

    with col4:
        if stats and stats.get('avg_hrv'):
            st.metric("Average HRV (SDNN)", f"{stats['avg_hrv']:.1f} ms")
        else:
            st.metric("Average HRV (SDNN)", "No data")

    st.divider()


except Exception as e:
    st.error(f"Error loading patient data: {e}")
    st.info("Please update your profile and upload recordings to see your summary.")
