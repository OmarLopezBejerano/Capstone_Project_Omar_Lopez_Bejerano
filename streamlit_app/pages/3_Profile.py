import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.database import save_user_profile, get_user_profile

st.set_page_config(page_title="Profile", layout="wide")

st.title("Profile Settings")

st.markdown("Manage your personal information and health data. This information can help contextualize your blood pressure readings.")

st.divider()

# Load existing profile, if available
try:
    profile = get_user_profile()
    if not profile:
        profile = {
            'name': '',
            'age': None,
            'sex': None,
            'height_cm': None,
            'weight_kg': None,
            'medical_conditions': '',
            'medications': '',
            'notes': ''
        }
except:
    profile = {
        'name': '',
        'age': None,
        'sex': None,
        'height_cm': None,
        'weight_kg': None,
        'medical_conditions': '',
        'medications': '',
        'notes': ''
    }

# Profile form
with st.form("profile_form"):
    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name", value=profile.get('name', ''))
        age = st.number_input("Age", min_value=1, max_value=120,value=profile.get('age', 30) if profile.get('age') else 30)

        # Handle sex selection with proper default
        sex_options = ["", "Male", "Female", "Other"]
        current_sex = profile.get('sex', '')
        sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
        sex = st.selectbox("Sex", sex_options, index=sex_index)

    with col2:
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0,value=float(profile.get('height_cm', 170.0)) if profile.get('height_cm') else 170.0,step=0.1)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0,value=float(profile.get('weight_kg', 70.0)) if profile.get('weight_kg') else 70.0,step=0.1)

        # Calculate BMI
        if height_cm and weight_kg:
            bmi = weight_kg / ((height_cm/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")

    st.divider()

    st.subheader("Medical Information")

    medical_conditions = st.text_area(
        "Medical Conditions",
        value=profile.get('medical_conditions', ''),
        help="List any relevant medical conditions (e.g., hypertension, diabetes)",
        height=100
    )

    medications = st.text_area(
        "Current Medications",
        value=profile.get('medications', ''),
        help="List current medications (especially BP medications)",
        height=100
    )

    st.divider()

    st.subheader("Additional Notes")

    notes = st.text_area(
        "Notes",
        value=profile.get('notes', ''),
        help="Any additional information you'd like to provide",
        height=150
    )

    st.markdown("---")

    # Submit button
    submitted = st.form_submit_button("Save Changes", type="primary", use_container_width=True)

    if submitted:
        try:
            # Save profile to database
            save_user_profile(
                name=name,
                age=age,
                sex=sex if sex else None,
                height_cm=height_cm,
                weight_kg=weight_kg,
                medical_conditions=medical_conditions,
                medications=medications,
                notes=notes
            )

            st.success("Profile saved successfully!")

        except Exception as e:
            st.error(f"Error saving profile: {e}")
