import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="TalentPulse | Employee Attrition Intelligence",
    page_icon="📊",
    layout="wide"
)

# 2. LOAD TRAINED MODEL

MODEL_PATH = Path(__file__).parent / "models" / "talentpulse_xgboost.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =====================================================
# 3. IBM DATASET VALUE MAPPINGS
# =====================================================

education_options = {
    "Below College": 1,
    "College": 2,
    "Bachelor": 3,
    "Master": 4,
    "Doctor": 5
}

satisfaction_options = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

work_life_options = {
    "Bad": 1,
    "Good": 2,
    "Better": 3,
    "Best": 4
}

performance_options = {
    "Excellent": 3,
    "Outstanding": 4
}


# =====================================================
# 4. APPLICATION HEADER
# =====================================================

st.title("📊 TalentPulse")
st.subheader("Employee Attrition Intelligence")

st.markdown(
    """
    **Turning Workforce Data into Predictive Insights**

    TalentPulse uses Machine Learning to estimate employee attrition
    probability based on employee demographics, job characteristics,
    compensation and work experience.

    Enter employee information below to generate a prediction.
    """
)

st.divider()


# =====================================================
# 5. EMPLOYEE INPUT FORM
# =====================================================

with st.form("employee_form"):

    st.header("Employee Information")

    # -------------------------------------------------
    # PERSONAL DETAILS
    # -------------------------------------------------

    st.subheader("👤 Personal Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        Age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=30
        )

        Gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        MaritalStatus = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )

    with col2:

        Education = education_options[
            st.selectbox(
                "Education",
                list(education_options.keys())
            )
        ]

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

    with col3:

        DistanceFromHome = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=29,
            value=5
        )

    st.divider()

    # -------------------------------------------------
    # JOB DETAILS
    # -------------------------------------------------

    st.subheader("💼 Job Details")

    col1, col2, col3 = st.columns(3)

    with col1:

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

    with col2:

        JobLevel = st.selectbox(
            "Job Level",
            [1, 2, 3, 4, 5]
        )

        JobInvolvement = satisfaction_options[
            st.selectbox(
                "Job Involvement",
                list(satisfaction_options.keys())
            )
        ]

    with col3:

        JobSatisfaction = satisfaction_options[
            st.selectbox(
                "Job Satisfaction",
                list(satisfaction_options.keys())
            )
        ]

        EnvironmentSatisfaction = satisfaction_options[
            st.selectbox(
                "Environment Satisfaction",
                list(satisfaction_options.keys())
            )
        ]

    st.divider()

    # -------------------------------------------------
    # SALARY AND EXPERIENCE
    # -------------------------------------------------

    st.subheader("💰 Salary & Work Experience")

    col1, col2, col3 = st.columns(3)

    with col1:

        MonthlyIncome = st.number_input(
            "Monthly Income",
            min_value=1009,
            max_value=19999,
            value=5000
        )

        MonthlyRate = st.number_input(
            "Monthly Rate",
            min_value=2094,
            max_value=26999,
            value=14000
        )

        DailyRate = st.number_input(
            "Daily Rate",
            min_value=102,
            max_value=1499,
            value=800
        )

        HourlyRate = st.number_input(
            "Hourly Rate",
            min_value=30,
            max_value=100,
            value=65
        )

    with col2:

        TotalWorkingYears = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=40,
            value=8
        )

        YearsAtCompany = st.number_input(
            "Years At Company",
            min_value=0,
            max_value=40,
            value=5
        )

        YearsInCurrentRole = st.number_input(
            "Years In Current Role",
            min_value=0,
            max_value=18,
            value=3
        )

    with col3:

        YearsWithCurrManager = st.number_input(
            "Years With Current Manager",
            min_value=0,
            max_value=17,
            value=3
        )

        YearsSinceLastPromotion = st.number_input(
            "Years Since Last Promotion",
            min_value=0,
            max_value=15,
            value=1
        )

        NumCompaniesWorked = st.number_input(
            "Previous Companies Worked",
            min_value=0,
            max_value=9,
            value=2
        )

    st.divider()

    # -------------------------------------------------
    # ADDITIONAL WORK INFORMATION
    # -------------------------------------------------

    st.subheader("📈 Additional Work Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        OverTime = st.selectbox(
            "Overtime",
            ["Yes", "No"]
        )

        StockOptionLevel = st.selectbox(
            "Stock Option Level",
            [0, 1, 2, 3]
        )

    with col2:

        PercentSalaryHike = st.number_input(
            "Percent Salary Hike",
            min_value=11,
            max_value=25,
            value=15
        )

        PerformanceRating = performance_options[
            st.selectbox(
                "Performance Rating",
                list(performance_options.keys())
            )
        ]

    with col3:

        RelationshipSatisfaction = satisfaction_options[
            st.selectbox(
                "Relationship Satisfaction",
                list(satisfaction_options.keys())
            )
        ]

        TrainingTimesLastYear = st.number_input(
            "Training Times Last Year",
            min_value=0,
            max_value=6,
            value=3
        )

        WorkLifeBalance = work_life_options[
            st.selectbox(
                "Work Life Balance",
                list(work_life_options.keys())
            )
        ]

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Employee Attrition",
        use_container_width=True,
        type="primary"
    )


# =====================================================
# 6. INPUT VALIDATION
# =====================================================

if submitted:

    validation_errors = []

    if YearsAtCompany > TotalWorkingYears:
        validation_errors.append(
            "Years at company cannot exceed total working years."
        )

    if YearsInCurrentRole > YearsAtCompany:
        validation_errors.append(
            "Years in current role cannot exceed years at company."
        )

    if YearsWithCurrManager > YearsAtCompany:
        validation_errors.append(
            "Years with current manager cannot exceed years at company."
        )

    if YearsSinceLastPromotion > YearsAtCompany:
        validation_errors.append(
            "Years since last promotion cannot exceed years at company."
        )

    if validation_errors:

        for error in validation_errors:
            st.error(error)

        st.stop()


    # =================================================
    # 7. CREATE EMPLOYEE DATAFRAME
    # =================================================

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


    # =================================================
    # 8. ENSURE CORRECT FEATURE ORDER
    # =================================================

    expected_features = [
        "Age",
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EnvironmentSatisfaction",
        "Gender",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    employee_data = employee_data[expected_features]


    # =================================================
    # 9. MODEL PREDICTION
    # =================================================

    try:

        prediction = model.predict(employee_data)[0]

        probabilities = model.predict_proba(employee_data)[0]

        classes = model.named_steps["classifier"].classes_

        yes_probability = probabilities[
            list(classes).index(1)
        ]


        # =============================================
        # DISPLAY RESULTS
        # =============================================

        st.divider()

        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error("⚠️ Predicted Attrition: YES")

            else:

                st.success("✅ Predicted Attrition: NO")

        with col2:

            st.metric(
                "Estimated Attrition Probability",
                f"{yes_probability * 100:.2f}%"
            )


        # =============================================
        # PROBABILITY VISUALIZATION
        # =============================================

        st.subheader("Attrition Probability")

        st.progress(float(yes_probability))

        st.caption(
            f"Estimated probability: {yes_probability * 100:.2f}%"
        )


        # =============================================
        # EMPLOYEE INPUT SUMMARY
        # =============================================

        with st.expander("View Submitted Employee Information"):

            st.dataframe(
                employee_data.T.rename(
                    columns={0: "Employee Information"}
                ),
                use_container_width=True
            )


    except Exception as e:

        st.error(f"Prediction failed: {e}")