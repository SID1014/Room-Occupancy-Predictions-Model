import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

st.header('Room Occupacy Detection')
st.title('Multivariate Model')

def multi_model():
    pipeline = joblib.load('final_model_robust.joblib')
    with open('training_columns.json', 'r') as f:
        columns = json.load(f)
    return pipeline, columns

multi_model_pipeline = multi_model()

s1_temp = st.slider("S1 Temperature (°C)", 20.0, 30.0, 25.0)
s1_light = st.slider("S1 Light (Lux)", 0, 500, 150)
s2_light = st.slider("S2 Light (Lux)", 0, 500, 150)
s3_light = st.slider("S3 Light (Lux)", 0, 500, 150)
s5_co2 = st.slider("S5 CO2 (ppm)", 300, 1500, 450)
s1_sound = st.slider("S1 Sound", 0.0, 1.5, 0.1)
s2_sound = st.slider("S2 Sound", 0.0, 1.5, 0.1)
s6_PIR = st.slider("S6 PIR",0,1,step=1)
s7_PIR = st.slider("S7 PIR",0,1,step=1)
hour = st.slider("Hour of the Day (0-23)", 0, 23, 10)

def feture_engineering():
    input_data = {
        
        'S1_Temp':s1_temp,
        'S2_Temp':25.54605884095172,
        'S3_Temp':25.0566205943331,
        'S4_Temp':25.75412479020634,
        'S1_Light':s1_light,
        'S2_Light':s2_light,
        'S3_Light':s3_light,
        'S1_Sound':s1_sound,
        'S2_Sound':s2_sound,
        'S3_Sound':0.15811926152631062,
        'S5_CO2': s5_co2,
        'S5_CO2_Slope':-0.0048300006835023225,
        'S6_PIR':s6_PIR,
        'S7_PIR':s7_PIR,
        'Day':21
    }
    input_data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    input_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    input_data['S1_Sound_lag'] = 0.168177510119459
    input_data['S5_CO2_window'] = 460.8648435186099
    input_df = pd.DataFrame([input_data])
    columns = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light',
       'S3_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S5_CO2',
       'S5_CO2_Slope', 'S6_PIR', 'S7_PIR', 'Day', 'hour_sin', 'hour_cos',
       'S1_Sound_lag', 'S5_CO2_window']
    return input_df[columns]

def final_predictions(custom_input):
    prediction = multi_model_pipeline.predict(custom_input)
    return prediction

if st.button('Predict Results'):
    predictions = final_predictions(feture_engineering())
    st.success(f"Occording to our model there are {predictions[0]} Occupants in room.")