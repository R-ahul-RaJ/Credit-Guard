import streamlit as st
import dill
import pandas as pd

# Load the model and pipeline
def load_model():
    with open('model/model_pipeline_object.joblib', 'rb') as file:
        data = dill.load(file)
    return data

data = load_model()
model = data["model"]
pipeline = data["pipeline"]

# Define the Streamlit app layout
st.set_page_config(
    page_title="🔮 Credit Card Default Prediction",
    page_icon="💳",
    layout="centered",
)

st.title("🔮 Credit Card Default Prediction")
st.sidebar.header("👤 User Inputs")


# Define the input fields for each feature
LIMIT_BAL = st.sidebar.number_input("💳 Credit Limit (LIMIT_BAL)", min_value=0)
SEX = st.sidebar.number_input("🚻 Gender (M=1 , F=2)", min_value=0)
EDUCATION = st.sidebar.number_input("📚Education Level (graduate=1 , university=2 , highschool=3)", min_value=0)
MARRIAGE = st.sidebar.number_input("💍 Marital Status (Married=1 Unmarried=2 others=3)", min_value=0)
AGE = st.sidebar.number_input("📆 Age (AGE)", min_value=0)
PAY_SEPT = st.sidebar.slider("💵 Payment Status in September (PAY_SEPT)", min_value=-3, max_value=8)
PAY_AUG = st.sidebar.slider("💵 Payment Status in August (PAY_AUG)", min_value=-3, max_value=8)
PAY_JUL = st.sidebar.slider("💵 Payment Status in July (PAY_JUL)", min_value=-3, max_value=8)
PAY_JUN = st.sidebar.slider("💵 Payment Status in June (PAY_JUN)", min_value=-3, max_value=8)
PAY_MAY = st.sidebar.slider("💵 Payment Status in May (PAY_MAY)", min_value=-3, max_value=8)
PAY_APR = st.sidebar.slider("💵 Payment Status in April (PAY_APR)", min_value=-3, max_value=8)
BILL_AMT_SEPT = st.sidebar.number_input("💵 Bill Amount in September (BILL_AMT_SEPT)", min_value=0)
BILL_AMT_AUG = st.sidebar.number_input("💵 Bill Amount in August (BILL_AMT_AUG)", min_value=0)
BILL_AMT_JUL = st.sidebar.number_input("💵 Bill Amount in July (BILL_AMT_JUL)", min_value=0)
BILL_AMT_JUN = st.sidebar.number_input("💵 Bill Amount in June (BILL_AMT_JUN)", min_value=0)
BILL_AMT_MAY = st.sidebar.number_input("💵 Bill Amount in May (BILL_AMT_MAY)", min_value=0)
BILL_AMT_APR = st.sidebar.number_input("💵 Bill Amount in April (BILL_AMT_APR)", min_value=0)
PAY_AMT_SEPT = st.sidebar.number_input("💵 Payment Amount in September (PAY_AMT_SEPT)", min_value=0)
PAY_AMT_AUG = st.sidebar.number_input("💵 Payment Amount in August (PAY_AMT_AUG)", min_value=0)
PAY_AMT_JUL = st.sidebar.number_input("💵 Payment Amount in July (PAY_AMT_JUL)", min_value=0)
PAY_AMT_JUN = st.sidebar.number_input("💵 Payment Amount in June (PAY_AMT_JUN)", min_value=0)
PAY_AMT_MAY = st.sidebar.number_input("💵 Payment Amount in May (PAY_AMT_MAY)", min_value=0)
PAY_AMT_APR = st.sidebar.number_input("💵 Payment Amount in April (PAY_AMT_APR)", min_value=0)

# Function to make predictions
def make_prediction():
    # Create a DataFrame from the user inputs
    user_inputs_df = pd.DataFrame({
        'LIMIT_BAL': [LIMIT_BAL],
        'SEX': [SEX],
        'EDUCATION': [EDUCATION],
        'MARRIAGE': [MARRIAGE],
        'AGE': [AGE],
        'PAY_SEPT': [PAY_SEPT],
        'PAY_AUG': [PAY_AUG],
        'PAY_JUL': [PAY_JUL],
        'PAY_JUN': [PAY_JUN],
        'PAY_MAY': [PAY_MAY],
        'PAY_APR': [PAY_APR],
        'BILL_AMT_SEPT': [BILL_AMT_SEPT],
        'BILL_AMT_AUG': [BILL_AMT_AUG],
        'BILL_AMT_JUL': [BILL_AMT_JUL],
        'BILL_AMT_JUN': [BILL_AMT_JUN],
        'BILL_AMT_MAY': [BILL_AMT_MAY],
        'BILL_AMT_APR': [BILL_AMT_APR],
        'PAY_AMT_SEPT': [PAY_AMT_SEPT],
        'PAY_AMT_AUG': [PAY_AMT_AUG],
        'PAY_AMT_JUL': [PAY_AMT_JUL],
        'PAY_AMT_JUN': [PAY_AMT_JUN],
        'PAY_AMT_MAY': [PAY_AMT_MAY],
        'PAY_AMT_APR': [PAY_AMT_APR]
    })

    # You may need to preprocess the input data using your pipeline
    if pipeline is not None:
        processed_inputs = pipeline.transform(user_inputs_df)
    else:
        processed_inputs = user_inputs_df  # If no preprocessing is needed

    prediction = model.predict(processed_inputs)
    return prediction[0]


if st.sidebar.button("🔮 Predict"):
    prediction = make_prediction()
    st.subheader("📊 Prediction Result:")
    if prediction == 0:
        st.success("✅ The customer is not expected to default on the payment.")
        st.image('images/not default.jpg', caption='Prediction 0 Image', use_column_width=True)
        st.balloons()
    elif prediction == 1:
        st.error("❌ The customer is expected to default on the payment.")
        st.image('images/default.jpg', caption='Prediction 1 Image', use_column_width=True)
        st.warning("💔 Please review the user inputs and consider taking appropriate action.")
