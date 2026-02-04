import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open('model.pk1', 'rb') as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction Dashboard")
st.markdown("---")
st.markdown("### 🎯 Enter customer information below to predict churn probability")

st.sidebar.title("📋 Instructions")
st.sidebar.markdown("""
**How to use this dashboard:**

1. 📝 Fill in all customer details in the form
2. 🔍 Review your selections 
3. 🚀 Click 'Predict Customer Churn' 
4. 📈 View the prediction results

**Features included:**
- 👤 Demographics
- 📞 Phone Services  
- 🌐 Internet Services
- 📄 Contract Details
- 💳 Payment Information
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Hover over labels for more information!")

st.markdown("### � Customer Information Form")
st.markdown("Fill in all the customer details below:")

with st.expander("📊 Basic Customer Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        SeniorCitizen = st.selectbox("👴 Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                                   help="Is the customer a senior citizen?")
        tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=100, value=1,
                               help="Number of months the customer has stayed with the company")
        
    with col2:
        MonthlyCharges = st.number_input("💰 Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0,
                                       help="The amount charged to the customer monthly")
        TotalCharges = st.number_input("💳 Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0,
                                     help="The total amount charged to the customer")

with st.expander("👥 Demographics", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_Male = st.selectbox("👤 Gender", ["Female", "Male"], index=0)
    with col2:
        Partner_Yes = st.selectbox("💑 Partner", ["No", "Yes"], index=0, help="Does the customer have a partner?")
    with col3:
        Dependents_Yes = st.selectbox("👶 Dependents", ["No", "Yes"], index=0, help="Does the customer have dependents?")

with st.expander("📞 Phone Services", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        PhoneService_Yes = st.selectbox("📱 Phone Service", ["No", "Yes"], index=1, 
                                      help="Does the customer have a phone service?")
    with col2:
        MultipleLines = st.selectbox("📞 Multiple Lines", ["No", "Yes", "No phone service"], index=0,
                                   help="Does the customer have multiple lines?")

with st.expander("🌐 Internet Services", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        InternetService = st.selectbox("🔌 Internet Service Type", ["No", "DSL", "Fiber optic"], index=1,
                                     help="What type of internet service does the customer have?")
        OnlineSecurity = st.selectbox("🔒 Online Security", ["No", "Yes", "No internet service"], index=0)
        OnlineBackup = st.selectbox("💾 Online Backup", ["No", "Yes", "No internet service"], index=0)
        
    with col2:
        DeviceProtection = st.selectbox("🛡️ Device Protection", ["No", "Yes", "No internet service"], index=0)
        TechSupport = st.selectbox("🔧 Tech Support", ["No", "Yes", "No internet service"], index=0)
        StreamingTV = st.selectbox("📺 Streaming TV", ["No", "Yes", "No internet service"], index=0)
        
    StreamingMovies = st.selectbox("🎬 Streaming Movies", ["No", "Yes", "No internet service"], index=0)

with st.expander("💳 Contract & Billing", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        Contract = st.selectbox("📋 Contract Type", ["Month-to-month", "One year", "Two year"], index=0,
                              help="What is the contract term of the customer?")
        PaperlessBilling_Yes = st.selectbox("📄 Paperless Billing", ["No", "Yes"], index=0,
                                          help="Does the customer have paperless billing?")
    with col2:
        PaymentMethod = st.selectbox("💰 Payment Method", 
                                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 
                                    index=0, help="How does the customer pay their bill?")

def convert_inputs():
    gender_Male_val = 1 if gender_Male == "Male" else 0
    
    Partner_Yes_val = 1 if Partner_Yes == "Yes" else 0
    
    Dependents_Yes_val = 1 if Dependents_Yes == "Yes" else 0
    
    PhoneService_Yes_val = 1 if PhoneService_Yes == "Yes" else 0
    
    MultipleLines_No = 1 if MultipleLines == "No" else 0
    MultipleLines_No_phone_service = 1 if MultipleLines == "No phone service" else 0
    MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0
    
    InternetService_DSL = 1 if InternetService == "DSL" else 0
    InternetService_Fiber_optic = 1 if InternetService == "Fiber optic" else 0
    InternetService_No = 1 if InternetService == "No" else 0
    
    OnlineSecurity_No = 1 if OnlineSecurity == "No" else 0
    OnlineSecurity_No_internet_service = 1 if OnlineSecurity == "No internet service" else 0
    
    OnlineBackup_No = 1 if OnlineBackup == "No" else 0
    OnlineBackup_No_internet_service = 1 if OnlineBackup == "No internet service" else 0
    
    DeviceProtection_No = 1 if DeviceProtection == "No" else 0
    DeviceProtection_No_internet_service = 1 if DeviceProtection == "No internet service" else 0
    
    TechSupport_No = 1 if TechSupport == "No" else 0
    TechSupport_No_internet_service = 1 if TechSupport == "No internet service" else 0
    
    StreamingTV_No = 1 if StreamingTV == "No" else 0
    StreamingTV_No_internet_service = 1 if StreamingTV == "No internet service" else 0
    
    StreamingMovies_No = 1 if StreamingMovies == "No" else 0
    StreamingMovies_No_internet_service = 1 if StreamingMovies == "No internet service" else 0
    
    Contract_Month_to_month = 1 if Contract == "Month-to-month" else 0
    Contract_One_year = 1 if Contract == "One year" else 0
    Contract_Two_year = 1 if Contract == "Two year" else 0
    
    PaperlessBilling_Yes_val = 1 if PaperlessBilling_Yes == "Yes" else 0
    
    PaymentMethod_Bank_transfer = 1 if PaymentMethod == "Bank transfer (automatic)" else 0
    PaymentMethod_Credit_card = 1 if PaymentMethod == "Credit card (automatic)" else 0
    PaymentMethod_Electronic_check = 1 if PaymentMethod == "Electronic check" else 0
    PaymentMethod_Mailed_check = 1 if PaymentMethod == "Mailed check" else 0
    
    return [
        SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
        gender_Male_val, Partner_Yes_val, Dependents_Yes_val, PhoneService_Yes_val,
        MultipleLines_No, MultipleLines_No_phone_service, MultipleLines_Yes,
        InternetService_DSL, InternetService_Fiber_optic, InternetService_No,
        OnlineSecurity_No, OnlineSecurity_No_internet_service,
        OnlineBackup_No, OnlineBackup_No_internet_service,
        DeviceProtection_No, DeviceProtection_No_internet_service,
        TechSupport_No, TechSupport_No_internet_service,
        StreamingTV_No, StreamingTV_No_internet_service,
        StreamingMovies_No, StreamingMovies_No_internet_service,
        Contract_Month_to_month, Contract_One_year, Contract_Two_year,
        PaperlessBilling_Yes_val,
        PaymentMethod_Bank_transfer, PaymentMethod_Credit_card,
        PaymentMethod_Electronic_check, PaymentMethod_Mailed_check
    ]

st.markdown("---")
st.markdown("### 🚀 Make Prediction")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True)

if predict_button:
    input_features = np.array([convert_inputs()])
    
    try:
        with st.spinner('🔄 Analyzing customer data...'):
            prediction = model.predict(input_features)
            prediction_proba = model.predict_proba(input_features)
        
        st.success("✅ Prediction completed successfully!")
        
        churn_probability = prediction_proba[0][1] * 100
        stay_probability = prediction_proba[0][0] * 100
        
        st.markdown("### 📈 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction[0] == 1:
                st.error("🚨 **HIGH CHURN RISK**")
                st.markdown(f"<h2 style='text-align: center; color: red;'>{churn_probability:.1f}% Likely to Churn</h2>", 
                           unsafe_allow_html=True)
            else:
                st.success("✅ **LOW CHURN RISK**")
                st.markdown(f"<h2 style='text-align: center; color: green;'>{stay_probability:.1f}% Likely to Stay</h2>", 
                           unsafe_allow_html=True)
        
        st.markdown("### 📊 Detailed Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏠 Stay Probability",
                value=f"{stay_probability:.1f}%",
                delta=f"{stay_probability - 50:.1f}%" if stay_probability > 50 else f"{stay_probability - 50:.1f}%"
            )
        
        with col2:
            st.metric(
                label="🚪 Churn Probability", 
                value=f"{churn_probability:.1f}%",
                delta=f"{churn_probability - 50:.1f}%" if churn_probability > 50 else f"{churn_probability - 50:.1f}%"
            )
        
        with col3:
            risk_level = "HIGH" if churn_probability > 70 else "MEDIUM" if churn_probability > 40 else "LOW"
            st.metric("⚠️ Risk Level", risk_level)
        
        with col4:
            recommendation = "Immediate Action" if churn_probability > 70 else "Monitor Closely" if churn_probability > 40 else "Maintain Service"
            st.metric("💡 Recommendation", recommendation)
        st.markdown("### 📊 Visual Probability")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Stay Probability**")
            st.progress(stay_probability/100)
            
        with col2:
            st.markdown("**Churn Probability**") 
            st.progress(churn_probability/100)
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("💡 Please check that all required features are provided and the model file is accessible.")
st.markdown("---")