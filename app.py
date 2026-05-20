
import streamlit as st
import joblib
import numpy as np

model          = joblib.load('career_model.pkl')
scaler         = joblib.load('scaler.pkl')
target_encoder = joblib.load('target_encoder.pkl')
cat_encoders   = joblib.load('cat_encoders.pkl')

# DEBUG — remove after fixing
st.write("Keys in cat_encoders:", list(cat_encoders.keys()))
