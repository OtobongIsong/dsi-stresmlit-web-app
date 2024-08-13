# Import libraries
import streamlit as st
import pandas as pd
import joblib

#Load our model pipeline object

model = joblib.load("model.joblib")


# Add Title and Instructions

st.title("Purchase Prediction Model")
st.subheader("Enter customer informaton and submit for likelihood to purchase")

# age input form
age = st.number_input(
     label = "01. Enter the customer's age",
     min_value = 18,
     max_value = 120,
     value = 35
     )
# Age input using slider

#age = st.slider("Age", min_value=18, max_value=120, value=35)


# Input validation
if age < 18 or age > 120:
    st.error("Please enter a valid age between 18 and 120.")


# gender input form
gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ['M', 'F']
    )

# credit score input form
credit_score = st.number_input(
     label = "03. Enter the customer's credit_score",
     min_value = 0,
     max_value = 1000,
     value = 500
     )


# Submit Input to model

if st.button("Submit for prediction"):
    # store our data in a datarame for prediction
    new_data = pd.DataFrame({"age": [age], "gender": [gender], "credit_score": [credit_score]})
    
    # Apply model pipeline to the model data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    #output prediction
    st.subheader(f"Based on this customer attribute, our model predict a probability of {pred_proba:.0%}")
    







