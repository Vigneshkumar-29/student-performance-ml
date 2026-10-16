"""
app.py - Streamlit frontend for the Student Performance Prediction System.
Start it with:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, PARTICIPATION_ORDER
# Importing this makes sure Python can find add_features, which is stored inside the saved model.
from src.features import add_features  # noqa: F401

st.set_page_config(page_title="Student Performance Prediction System", page_icon="🎓")


# ---------------------------------------------------------------------------
# Load the trained model ONCE (cache_resource keeps it in memory between clicks)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found. Please train the model first:  `python -m src.train`")
    st.stop()

bundle = load_model()
pipeline = bundle["pipeline"]


# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------
st.title("🎓 Student Performance Prediction System")
st.write("Enter a student's details in the sidebar, then click **Predict** to see "
         "whether the student is likely to **Pass** or **Fail**.")

# ---------------------------------------------------------------------------
# Sidebar: student information
# The min/max values stop impossible numbers from being typed (basic validation).
# ---------------------------------------------------------------------------
st.sidebar.header("Student information")
study_hours = st.sidebar.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
previous_score = st.sidebar.slider("Previous exam score", 0, 100, 60)
assignment_score = st.sidebar.slider("Assignment score", 0, 100, 65)
internal_score = st.sidebar.slider("Internal assessment score", 0, 100, 62)
sleep_hours = st.sidebar.slider("Sleep hours per night", 3.0, 12.0, 7.0, 0.5)
class_participation = st.sidebar.selectbox("Class participation", PARTICIPATION_ORDER, index=1)
previous_failed_subjects = st.sidebar.number_input(
    "Previous failed subjects", min_value=0, max_value=10, value=0, step=1)


# ---------------------------------------------------------------------------
# Extra validation: combinations that make no sense
# ---------------------------------------------------------------------------
def find_input_problems():
    problems = []
    if study_hours + sleep_hours > 22:
        problems.append("Study hours + sleep hours is more than 22 per day. Please re-check these values.")
    return problems


def explain(probability_pass):
    """Turn the inputs into a short, human-readable explanation."""
    risks, strengths = [], []

    if attendance < 75:
        risks.append(f"Attendance is low ({attendance}%, below the usual 75% requirement).")
    elif attendance >= 85:
        strengths.append(f"Attendance is strong ({attendance}%).")

    if previous_failed_subjects >= 2:
        risks.append(f"The student has failed {previous_failed_subjects} subjects before.")
    elif previous_failed_subjects == 0:
        strengths.append("No previously failed subjects.")

    average_score = (previous_score + assignment_score + internal_score) / 3
    if average_score < 55:
        risks.append(f"Average academic score is low ({average_score:.0f}/100).")
    elif average_score >= 70:
        strengths.append(f"Average academic score is high ({average_score:.0f}/100).")

    if study_hours < 2.5:
        risks.append(f"Study time is low ({study_hours} hours/day).")
    elif study_hours >= 5:
        strengths.append(f"Good study routine ({study_hours} hours/day).")

    if class_participation == "Low":
        risks.append("Class participation is low.")
    elif class_participation == "High":
        strengths.append("Class participation is high.")

    return risks, strengths


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("Predict", type="primary"):
    problems = find_input_problems()
    if problems:
        for message in problems:
            st.error(message)
    else:
        # Put the inputs in a one-row table with the SAME column names used in training.
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "assignment_score": assignment_score,
            "internal_score": internal_score,
            "sleep_hours": sleep_hours,
            "previous_failed_subjects": previous_failed_subjects,
            "class_participation": class_participation,
        }])

        # The saved pipeline does ALL preprocessing automatically, then predicts.
        prediction = pipeline.predict(student)[0]
        pass_column = list(pipeline.classes_).index(1)
        probability_pass = pipeline.predict_proba(student)[0][pass_column]

        st.subheader("Result")
        if prediction == 1:
            st.success("Prediction: PASS ✅")
        else:
            st.error("Prediction: FAIL ❌")

        col1, col2 = st.columns(2)
        col1.metric("Probability of passing", f"{probability_pass:.0%}")
        col2.metric("Probability of failing", f"{1 - probability_pass:.0%}")
        st.progress(float(probability_pass), text="Chance of passing")

        # Plain-English explanation of the probability
        if probability_pass >= 0.80:
            st.write("The model is **very confident** this student will pass.")
        elif probability_pass >= 0.55:
            st.write("The student is **likely to pass**, but the margin is not large.")
        elif probability_pass >= 0.45:
            st.write("This is a **borderline case**. Small changes could flip the result.")
        elif probability_pass >= 0.20:
            st.write("The student is **likely to fail** without improvement.")
        else:
            st.write("The model is **very confident** this student is at high risk of failing.")

        risks, strengths = explain(probability_pass)
        left, right = st.columns(2)
        with left:
            st.markdown("**Positive factors**")
            for item in strengths or ["None stood out."]:
                st.write("• " + item)
        with right:
            st.markdown("**Risk factors**")
            for item in risks or ["None stood out."]:
                st.write("• " + item)

        st.caption("This is a statistical prediction based on past patterns, not a guarantee. "
                   "Use it to offer support early, not to judge a student.")
else:
    st.info("Adjust the values in the sidebar and click **Predict**.")

# ---------------------------------------------------------------------------
# About the model
# ---------------------------------------------------------------------------
with st.expander("About the model"):
    metrics = bundle["metrics"]
    st.write(f"**Algorithm:** {bundle['model_name']}")
    st.write(f"**Test accuracy:** {metrics['Accuracy']:.0%} | **Precision:** {metrics['Precision']:.0%} | "
             f"**Recall:** {metrics['Recall']:.0%} | **F1-score:** {metrics['F1-score']:.0%}")
    st.write("Trained on a synthetic (practice) dataset of 1,500 students.")
