import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------
# 1. Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="TalentPulse | Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# 2. Load Trained Model
# -----------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "talentpulse_xgboost.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -----------------------------------
# 3. Application Header
# -----------------------------------

st.title("📊 TalentPulse")
st.subheader("Employee Attrition Prediction & Workforce Analytics")

st.markdown("""
TalentPulse uses Machine Learning to estimate employee attrition risk
based on employee demographics, job characteristics and work experience.

Enter employee information below to generate a prediction.
""")

st.divider()

# -----------------------------------
# 4. Employee Input Form
# -----------------------------------

with st.form("employee_form"):

    st.subheader("Employee Information")

    col1, col2, col3 = st.columns(3)

    # Demographics
    with col1:
        st.markdown("### Personal Details")

        Age = st.number_input("Age", 18, 60, 30)

        Gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        MaritalStatus = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )

        Education = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5]
        )

        EducationField = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )

        DistanceFromHome = st.number_input(
            "Distance From Home",
            1, 29, 5
        )

    # Job Details
    with col2:
        st.markdown("### Job Details")

        Department = st.selectbox(
            "Department",
            [
                "Sales",
                "Research & Development",
                "Human Resources"
            ]
        )

        JobRole = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )

        BusinessTravel = st.selectbox(
            "Business Travel",
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel"
            ]
        )

        JobLevel = st.selectbox(
            "Job Level",
            [1, 2, 3, 4, 5]
        )

        JobInvolvement = st.selectbox(
            "Job Involvement",
            [1, 2, 3, 4]
        )

        JobSatisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4]
        )

        EnvironmentSatisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4]
        )

    # Salary and Experience
    with col3:
        st.markdown("### Salary & Experience")

        MonthlyIncome = st.number_input(
            "Monthly Income",
            1009, 19999, 5000
        )

        MonthlyRate = st.number_input(
            "Monthly Rate",
            2094, 26999, 14000
        )

        DailyRate = st.number_input(
            "Daily Rate",
            102, 1499, 800
        )

        HourlyRate = st.number_input(
            "Hourly Rate",
            30, 100, 65
        )

        TotalWorkingYears = st.number_input(
            "Total Working Years",
            0, 40, 8
        )

        YearsAtCompany = st.number_input(
            "Years At Company",
            0, 40, 5
        )

        YearsInCurrentRole = st.number_input(
            "Years In Current Role",
            0, 18, 3
        )

        YearsWithCurrManager = st.number_input(
            "Years With Current Manager",
            0, 17, 3
        )

        YearsSinceLastPromotion = st.number_input(
            "Years Since Last Promotion",
            0, 15, 1
        )

    st.divider()

    st.subheader("Additional Work Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        OverTime = st.selectbox(
            "OverTime",
            ["Yes", "No"]
        )

        NumCompaniesWorked = st.number_input(
            "Number of Previous Companies",
            0, 9, 2
        )

        StockOptionLevel = st.selectbox(
            "Stock Option Level",
            [0, 1, 2, 3]
        )

    with col5:
        PercentSalaryHike = st.number_input(
            "Percent Salary Hike",
            11, 25, 15
        )

        PerformanceRating = st.selectbox(
            "Performance Rating",
            [3, 4]
        )

        RelationshipSatisfaction = st.selectbox(
            "Relationship Satisfaction",
            [1, 2, 3, 4]
        )

    with col6:
        TrainingTimesLastYear = st.number_input(
            "Training Times Last Year",
            0, 6, 3
        )

        WorkLifeBalance = st.selectbox(
            "Work Life Balance",
            [1, 2, 3, 4]
        )

    submitted = st.form_submit_button(
        "Predict Employee Attrition",
        use_container_width=True
    )

# -----------------------------------
# 5. Prediction
# -----------------------------------

if submitted:

    employee_data = pd.DataFrame([{
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }])

    try:

        prediction = model.predict(employee_data)[0]

        probabilities = model.predict_proba(employee_data)[0]

        classes = model.named_steps["classifier"].classes_

        yes_probability = probabilities[
            list(classes).index(1)
        ]

        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("⚠️ Predicted Attrition: Yes")
            else:
                st.success("✅ Predicted Attrition: No")

        with col2:

            st.metric(
                "Estimated Attrition Probability",
                f"{yes_probability * 100:.2f}%"
            )

        st.progress(float(yes_probability))

        st.info("""
        This prediction is based on patterns learned from historical
        employee data. It is an estimate, not a guarantee of future
        employee behaviour. It should not be used as the sole basis
        for employment decisions.
        """)

    except Exception as e:

        st.error(f"Prediction failed: {e}")