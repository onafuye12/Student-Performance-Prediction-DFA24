

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# Load pickled encoders and scaler

model = pickle.load(open('Student_Perfromance_model.pkl', 'rb'))
encoder = pickle.load(open('health_status_encoder.pkl', 'rb'))
scaler = pickle.load(open('minmax_scaler.pkl', 'rb'))


# Page title
st.title('Student Performance Prediction')

# Collect user inputs
st.header('Enter Student Data')

# Define the input fields for the Streamlit app
def user_input_features():
    sex = st.selectbox('Sex', ['M', 'F'])
    age = st.slider('Age', 10, 20, 15)
    famsize = st.selectbox('Family Size', ['LE3', 'GT3'])  # Ordinal
    Pstatus = st.selectbox('Parental Status', ['T', 'A'])
    Medu = st.selectbox('Mother\'s Education', ['No schooling', 'SSCE', 'Bachelor', 'Masters', 'PhD'])  # Ordinal
    Fedu = st.selectbox('Father\'s Education', ['No schooling', 'SSCE', 'Bachelor', 'Masters', 'PhD'])  # Ordinal
    SS1_result_avg = st.number_input('SS1 result average',0, 100)
    SS2_result_avg = st.number_input('SS2 result average',0, 100)
    SS3_mock_result = st.selectbox('SS3_mock_result', ['Pass', 'Fail'])
    absences = st.number_input('Absences', 0, 100)
    studytime = st.slider('Weekly Study Time (hours)', 1, 5, 4)
    extra_tutoring = st.selectbox('Extra Tutoring', ['yes', 'no'])
    school_support = st.selectbox('School Support', ['yes', 'no'])
    internet_access = st.selectbox('Internet Access', ['yes', 'no'])
    CBT_preparation = st.selectbox('CBT Preparation', ['yes', 'no'])
    health_status = st.selectbox('Health Status', ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good'])  # Ordinal
    wants_higher_education = st.selectbox('Wants Higher Education', ['yes', 'no'])
    wants_trade = st.selectbox('Wants Trade', ['yes', 'no'])

# Prepare input features for scaling and prediction
    input_data = pd.DataFrame([{
        'sex': sex,
        'age': age,
        'famsize': famsize,
        'Pstatus': Pstatus,
        'Medu': Medu,
        'Fedu': Fedu,
        'SS1_result_avg': SS1_result_avg,
        'SS2_result_avg' : SS2_result_avg,
        'SS3_mock_result' : SS3_mock_result,
        'absences': absences,
        'studytime': studytime,
        'extra_tutoring': extra_tutoring,
        'school_support': school_support,
        'internet_access': internet_access,
        'CBT_preparation': CBT_preparation,
        'health_status': health_status,
        'wants_higher_education': wants_higher_education,
        'wants_trade': wants_trade
    }])

    features = pd.DataFrame(input_data, index=[0])
    return features
# Ordinal encoding mappings

frame =  user_input_features()
def preprocess_features(df):
    # Manually define the correct hierarchical order for ordinal variables
    # 'Medu', 'Fedu': 'No schooling' -> 'SSCE' -> 'Bachelor' -> 'Masters' -> 'PhD'
    medu_fedu_order = ['No schooling', 'SSCE', 'Bachelor', 'Masters', 'PhD']

    # 'health_status': 'Very Poor' -> 'Poor' -> 'Average' -> 'Good' -> 'Very Good'
    health_status_order = ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good']

    # 'famsize': 'LE3' -> 'GT3' (LE3 means ≤ 3 members, GT3 means > 3 members)
    famsize_order = ['LE3', 'GT3']

    #target value: 
    CBT_outcome_order = ['High', 'Low']
    # Ordinal columns with specified order
    ordinal_cols = {
        'Medu': medu_fedu_order,
        'Fedu': medu_fedu_order,
        'health_status': health_status_order,
        'famsize': famsize_order
    }

    # Perform label encoding for the ordinal columns with a custom order
    for col, order in ordinal_cols.items():
        df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes

    # Columns that will be label encoded normally (no specific hierarchy)
    cat1 = ['sex', 'Pstatus', 'SS3_mock_result', 'extra_tutoring',
                         'school_support', 'internet_access', 'CBT_preparation',
                         'wants_higher_education', 'wants_trade']
# Encode categorical features
    encoded_cats = encoder.transform(df[cat1]).toarray()
    enc_data = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat1))
    df = df.join(enc_data)

    df.drop(cat1, axis=1, inplace=True)


    # Scale the input data
    columns_to_scale = ['age','Medu','Fedu', 'health_status','SS1_result_avg', 'SS2_result_avg', 'absences', 'studytime']

   # Apply the scaler to the numerical columns
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    
    return df


# Display the input features
st.write(frame)
 # Preprocess the input features
frame2 = preprocess_features(frame)


# Predict the outcome using the loaded model
if st.button('Predict'):
    prediction = model.predict(frame2)
    st.subheader('Prediction Result')
    st.write('High' if prediction[0] == 1 else 'Low')
