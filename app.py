import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

st.header("Room Occupancy Prediction Model")



tab_prediction, tab_insights = st.tabs(["Model Prediction Demo", "Project Insights"])

with tab_prediction:
    model_to_use =  st.selectbox("Which Model to Use",['Multivariate','Binary'])


    st.title(f'{model_to_use} Model')

    def multi_model():
        pipeline = joblib.load('Multi_Modle_Pipeline.joblib')
        return pipeline
    def bin_model():
        pipeline = joblib.load('Binary_Model_Pipeline.joblib')
        return pipeline

    multi_model_pipeline = multi_model()
    bin_model_pipeline = bin_model()

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

    def final_predictions(custom_input,model):
        prediction = model.predict(custom_input)
        return prediction

    if model_to_use == 'Multivariate':
        if st.button('Predict Results'):
            predictions = final_predictions(feture_engineering(),multi_model_pipeline)
            st.success(f"Occording to our model there are {predictions[0]} Occupants in room.")
    else:
        if st.button('Predict Results'):
            predictions = final_predictions(feture_engineering(),bin_model_pipeline)
            if predictions[0] == 1:
                result = 'Present'
            else :
                result = 'absent'
            st.success(f"Occording to our model Occupants are {result} in room.")
with tab_insights:
    st.header('Data insights and visualisations')
    st.title('Introduction :')
    st.text('This project is based on Room Occupancy datset from UCI ML Repositor\n It includes data from various  sensors present in the room to detect how' \
    'many occupants in the room.The max number Occupants seems to be 4 and minimum to be 0.\n' \
    'This Confirms that this is a Classification Problem.')


    st.title("Data Preparation :")
    st.text('There are two main features Date and Time which are representing that this is a time-series data' \
    ". On futher feature Engineering We have added three new features 'Day','Hour_cos' and 'Hour_sin'. 'Hour_cos' and "
    "'Hour_sin' represts hour but in cyclic matter so that model understands that hour 23 and hour 0 are very close to each other." \
    " We also added the some lag and rolling window feature so that model can learn from the past as this is a time series data model should learn from past also.\n" \
    "This is the correaltion matrix of dataset after feature Engineering")

    st.image('Project Insight images/dataset-correlation-matrix.png',caption='Dataset Correlation Matrix')

    st.text('Wecan see that some features are highly correlated while some are not.We drop to features here who ' \
    'which represented some sensor data which was correlated below 0.5.We do retain engineered as even if they are not highly correlated they' \
    ' can be use full in tree models.')




    st.title('Preprocessing :')
    st.text('Before moving further we take a look of various scales of data.')
    st.image('Project Insight images/dataset-histograms.png',caption='Histograms of various features.(count vs Scale)')
    st.text('It is clear that some features are binary while most of them are continuos.Also we can clearly see that' \
    ' some features are on a very different scale than others thats because this is a sensor data so this data is in different units' \
    '. This can affect our models predictions so we use standard scaler and create a preprocesssing pipeline.')




    st.title('Model Training :')
    st.text('Then we create train and test set with 70:30 ratio. We use four models for primary evaluation')
    
    st.subheader("Model Selection")

    st.markdown("Four baseline models were evaluated to select the most promising candidate for hyperparameter tuning. Performance was measured using the weighted F1-score.")
    
    baseline_scores = {
    'Model': ['RandomForest', 'KNeighbors', 'SGDClassifier', 'SVC'],
    'Training F1-Score': [0.9961, 0.9957, 0.9949, 0.9969],
    'Testing F1-Score': [0.8885, 0.8466, 0.4802, 0.4920]
    }

    scores_df = pd.DataFrame(baseline_scores)
    st.dataframe(scores_df)

    st.text('We find two promising models RandomForestClassifier and KNeighborsClassifer. We use SMOTE as we can see from histograms' \
    'that there is huge class imbalace we also add our preprocessing pipeline here.Then we use Random search CV and find best parameter for' \
    ' both classifiers.')

    baseline_scores = {
    'Model': ['RandomForest', 'KNeighbors'],
    'Training F1-Score': [0.9984, 0.9977],
    'Testing F1-Score': [0.8987, 0.8462]
    }
    scores_df = pd.DataFrame(baseline_scores)
    st.dataframe(scores_df)

    st.text('We find that RandomForest is doing better so we select it as our final model.\n Following are the importances placed by' \
    ' RandomForestClassifier on the features.')

    st.image('Project Insight images/Top-15-features.png',caption='Top 15 features from RandomForestClassifier')
    st.text('We can see that those engineered time series features do have a impact on our model')

    st.title('The Problem:')
    st.image('Project Insight images/multivariate-model-cm.png',caption='The confusion Matrix of Multivariate Model on testing data')
    st.text('We can see that there is no example of 1 person present in room in testing data.\n' \
    'Tha data was in a time series format so this is normal but we can see that due to a huge class imbalance model seems to be' \
    'tilted towards class zero.')
    
    st.title('The Solution:')

    st.text("To deal with this specific problem and considering that our dataset comes from a IoT system' \
    'Designed to switch lights on and off we can make this multivariate model into a binary model.We made the target class as a binary class that is if absent 0 else 1.\n" \
    "Then we again used RandomForestClassifier in our Pipeline. This time also we used random search CV to hypertune parameters.")
    st.image('Project Insight images/binary-model-cm.png',caption='Confusion Matrix of Binary Classifier')
    
    baseline_scores = {
    'Model': ['RandomForest'],
    'Training F1-Score': [0.9995],
    'Testing F1-Score': [0.9266]
    }
    scores_df = pd.DataFrame(baseline_scores)
    st.dataframe(scores_df)
    st.text("We can clearly see that model has imporved considerably.This is also beacuse we have tweaked the threshold a littile to fix the recall-precision tradeoff." )