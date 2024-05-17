import streamlit as st
import joblib
import pandas as pd

# Load the trained model
xgb_model = joblib.load('xgb_model.pkl')

# Define the input features
input_features = ['Age', 'Sex', 'Height', 'Weight', 'BMI', 'Body fat', 
                  'Hand grip dominant hand', 'Hand grip non-dominant hand', 
                  'Wrist circumference dominant', 'Wrist circumference non-dominant', 
                  'Serum T3', 'Serum T4', 'TSH']


# Calculate BMI function
def calculate_bmi(height, weight):
    if height > 0 and weight > 0:
        return round(weight / ((height / 100) ** 2), 2)
    else:
        return 0
    
st.title('Hypothyroidism Prediction App')


# Create input widgets for each feature
inputs = {}
for feature in input_features:
    if feature == 'Sex':
        inputs[feature] = st.radio(f'{feature}:', ['male', 'female'])
    elif feature == 'BMI':
        inputs[feature] = calculate_bmi(inputs['Height'], inputs['Weight']) 
    else:
        inputs[feature] = st.number_input(f'{feature}:')

#inputs['BMI'] = calculate_bmi(inputs['Height'], inputs['Weight'])

# Submit button
if st.button('Submit'):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs])



    input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})


    cols_to_exclude = ['Hand grip dominant hand', 'Hand grip non-dominant hand', 
                   'Wrist circumference dominant', 'Wrist circumference non-dominant']

    # Make prediction
    prediction = xgb_model.predict(input_df.drop(columns=cols_to_exclude))

    # Display prediction
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.error('The person is predicted to have Hypothyroidism.')
    else:
        st.success('The person is predicted to be Normal.')


