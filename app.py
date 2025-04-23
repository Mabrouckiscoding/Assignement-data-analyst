import streamlit as st
import pandas as pd
from datetime import datetime

# Load and prepare training data
df = pd.read_csv(r"Assignement-data-analyst/healthcare_dataset.csv")

# Train simple rule-based model
risk_model = df.groupby(['Gender', 'Blood Group Type'])['Medical Condition'] \
               .agg(lambda x: x.value_counts().idxmax()) \
               .reset_index()

def predict_condition(gender, blood_group):
    result = risk_model[
        (risk_model['Gender'] == gender) & 
        (risk_model['Blood Group Type'] == blood_group)
    ]
    if not result.empty:
        return result['Medical Condition'].values[0]
    else:
        return "Not enough data"

# --- Streamlit UI ---
st.title("Health Risk Predictor")

st.markdown("Enter your information to find out what medical condition you're most at risk for:")

# Web form
with st.form("health_form"):
    name = st.text_input("Name")
    gender = st.selectbox("Gender", ["Male", "Female"])
    blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    submit = st.form_submit_button("Check Risk")

# On form submission
if submit:
    condition = predict_condition(gender, blood_group)
    st.success(f"Hi {name}, based on your profile, you're most at risk for: **{condition}**")

    # Save to CSV
    new_entry = {
        "Timestamp": datetime.now().isoformat(),
        "Name": name,
        "Gender": gender,
        "Blood Group": blood_group,
        "Predicted Condition": condition
    }

    try:
        history_df = pd.read_csv(r"Assignement-data-analyst\user_inputs.csv")
        history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
    except FileNotFoundError:
        history_df = pd.DataFrame([new_entry])

    history_df.to_csv(r"Assignement-data-analyst\user_inputs.csv", index=False)
    st.info("Your data has been saved securely.")