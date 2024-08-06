import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Load and encode data for consistent encoding
df = pd.read_csv('data.csv')  # Load your data to fit encoders
le_dict = {}

# Label encode object columns
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

# Streamlit app title
st.title('Loan Prediction App')

# User inputs for the features
Gender = st.selectbox('Gender', le_dict['Gender'].classes_)
Married = st.selectbox('Married', le_dict['Married'].classes_)
Dependents = st.selectbox('Dependents', le_dict['Dependents'].classes_)
Education = st.selectbox('Education', le_dict['Education'].classes_)
Self_Employed = st.selectbox('Self_Employed', le_dict['Self_Employed'].classes_)
ApplicantIncome = st.number_input('ApplicantIncome', min_value=0)
CoapplicantIncome = st.number_input('CoapplicantIncome', min_value=0)
LoanAmount = st.number_input('LoanAmount', min_value=0)
Loan_Amount_Term = st.number_input('Loan_Amount_Term', min_value=0)
Credit_History = st.selectbox('Credit_History', [0.0, 1.0])
Property_Area = st.selectbox('Property_Area', le_dict['Property_Area'].classes_)

# Prepare the input data
input_data = pd.DataFrame({
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'Education': [Education],
    'Self_Employed': [Self_Employed],
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Property_Area': [Property_Area]
})

# Apply the same Label Encoding as in the training phase
for column in input_data.select_dtypes(include='object').columns:
    input_data[column] = le_dict[column].transform(input_data[column])

# Predict the loan status
prediction = model.predict(input_data)

# Display the prediction result
if st.button('Predict'):
    if prediction[0] == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Not Approved')
