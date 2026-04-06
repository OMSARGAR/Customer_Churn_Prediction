import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Load model & encoders
@st.cache_resource
def load_model():
    try:
        model = joblib.load("customer_churn_model.pkl")
        encoders = joblib.load("churn_encoders.pkl")
        return model, encoders
    except:
        return None, None

model, encoders = load_model()

st.title("Customer Churn Prediction")
st.markdown("Predict customer churn with Machine Learning")
st.markdown("---")

if model is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method",
                                     ["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"])

        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)

    st.markdown("---")

    if st.button("Predict Churn"):

        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_data])

        # ✅ Binary encoding
        input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

        # ✅ IMPORTANT FIX: Apply encoders correctly
        for col, encoder in encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except:
                    pass  # avoid crash if already numeric

        # ✅ Ensure all columns are numeric
        input_df = input_df.apply(pd.to_numeric)

        # Prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("WILL CHURN")
            else:
                st.success("WILL STAY")

        with col2:
            st.metric("Confidence", f"{max(proba) * 100:.1f}%")

        with col3:
            risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            st.metric("Risk Level", risk)

        # Chart
        fig = go.Figure(data=[
            go.Bar(name='No Churn', x=['Probability'], y=[proba[0]]),
            go.Bar(name='Churn', x=['Probability'], y=[proba[1]])
        ])
        fig.update_layout(barmode='group')
        st.plotly_chart(fig)

else:
    st.error("Model files not found")
