import streamlit as st
import joblib

st.title("Model Loading Test")

try:
    # Try to load the simple model
    simple_model = joblib.load('simple_model.joblib')
    st.success("Success! The simple_model.joblib file loaded correctly.")
    st.write(simple_model)
except Exception as e:
    st.error("Failed to load simple_model.joblib.")
    st.exception(e)

st.divider()

st.title("Original Model Test")

try:
    # Now, try to load your original, complex model
    original_model = joblib.load('Mulit_Modle_Pipeline.joblib') # Make sure filename is correct
    st.success("Success! The original pipeline loaded correctly.")
    st.write(original_model)
except Exception as e:
    st.error("Failed to load the original pipeline.")
    st.exception(e)