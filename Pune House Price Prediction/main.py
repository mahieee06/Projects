import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the data and model
data_path = 'Pune house data.csv'
model_path = 'Pune_Price_Prediction.pkl'

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv(data_path)
    return data

# Load the trained model using joblib
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load(model_path)
    return model

# Load the dataset and model
df = load_data()
model = load_model()

# Function to prepare input based on the model's expected features
def prepare_input(location, sqft, bath, bhk):
    # Create a dictionary of input values
    input_data = {
        'total_sqft': sqft,
        'bath': bath,
        'bhk': bhk
    }
    
    # One-hot encode location using pd.get_dummies
    location_dummies = pd.get_dummies(df['site_location'])
    
    # Create a DataFrame with input values
    input_df = pd.DataFrame([input_data])

    # Add location columns from the original data
    input_df = pd.concat([input_df, pd.DataFrame([np.zeros(len(location_dummies.columns))], columns=location_dummies.columns)], axis=1)
    
    # Set the selected location column to 1
    if location in input_df.columns:
        input_df[location] = 1

    # Ensure columns match the order of the training features
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    return input_df

# Function to predict price
def predict_price(location, sqft, bath, bhk):    
    input_df = prepare_input(location, sqft, bath, bhk)
    return model.predict(input_df)[0]

# Streamlit app title
st.title("Pune House Price Prediction App")

# User input fields for prediction
st.subheader("Enter the house details:")
location = st.selectbox("Location", df['site_location'].unique())
sqft = st.number_input("Total Area (in sq. ft)", min_value=300, max_value=5000, value=1000)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=5, value=2)

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(location, sqft, bath, bhk)
    st.write(f"Predicted House Price: ₹ {predicted_price:,.2f}")

# Show the raw dataset for reference
st.subheader("Dataset Sample")
st.write(df.head())
