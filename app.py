import streamlit as st
import joblib
import numpy as np

model          = joblib.load('career_model.pkl')
scaler         = joblib.load('scaler.pkl')
target_encoder = joblib.load('target_encoder.pkl')
cat_encoders   = joblib.load('cat_encoders.pkl')

st.set_page_config(page_title="Career Path Predictor", page_icon="🎯", layout="wide")
st.title("🎯 Career Path Prediction System")
st.markdown("Fill in your details below to get a personalized career recommendation.")

st.subheader("📊 Skills & Ratings")
col1, col2, col3 = st.columns(3)
with col1:
    logical    = st.slider("Logical Quotient Rating", 1, 10, 5)
    coding     = st.slider("Coding Skills Rating", 1, 10, 5)
with col2:
    speaking   = st.slider("Public Speaking Points", 1, 10, 5)
    hackathons = st.number_input("Hackathons Participated", 0, 20, 1)
with col3:
    memory = st.selectbox("Memory Capability Score", ["poor", "medium", "excellent"])

st.subheader("📚 Interests")
col4, col5 = st.columns(2)
with col4:
    subject     = st.selectbox("Interested Subject",
                    cat_encoders["interested_subjects"].classes_)
    career_area = st.selectbox("Interested Career Area",
                    cat_encoders["interested_career_area"].classes_)
with col5:
    company = st.selectbox("Type of Company",
                cat_encoders["type_of_company_want_to_settle_in?"].classes_)
    books   = st.selectbox("Interested Type of Books",
                cat_encoders["interested_type_of_books"].classes_)

st.subheader("🙋 About You")
col6, col7, col8 = st.columns(3)
with col6:
    self_learn = st.radio("Self Learner?",       ["Yes","No"], horizontal=True)
    extra_crs  = st.radio("Extra Courses?",      ["Yes","No"], horizontal=True)
    certs      = st.radio("Certifications?",     ["Yes","No"], horizontal=True)
with col7:
    workshops_ = st.radio("Workshops?",          ["Yes","No"], horizontal=True)
    seniors    = st.radio("Took Senior Input?",  ["Yes","No"], horizontal=True)
    teams      = st.radio("Worked in Teams?",    ["Yes","No"], horizontal=True)
with col8:
    introvert   = st.radio("Introvert?",         ["Yes","No"], horizontal=True)
    worker_type = st.radio("Worker Type?",       ["smart worker","hard worker"], horizontal=True)
    mgmt_tech   = st.radio("Preference?",        ["Management","Technical"], horizontal=True)

if st.button("🔮 Predict My Career", use_container_width=True):
    ordinal_map = {"poor": 1, "medium": 2, "excellent": 3}
    yn = lambda v: 1 if v == "Yes" else 0

    features = np.array([[
        logical, hackathons, coding, speaking,
        ordinal_map[memory],
        yn(self_learn), yn(extra_crs), yn(certs), yn(workshops_),
        yn(seniors), yn(teams), yn(introvert),
        1 if worker_type == "smart worker" else 0,
        1 if mgmt_tech   == "Management"   else 0,
        cat_encoders["interested_subjects"].transform([subject])[0],
        cat_encoders["interested_career_area"].transform([career_area])[0],
        cat_encoders["type_of_company_want_to_settle_in?"].transform([company])[0],
        cat_encoders["interested_type_of_books"].transform([books])[0]
    ]])

    scaled = scaler.transform(features)
    pred   = model.predict(scaled)[0]
    role   = target_encoder.inverse_transform([pred])[0]
    proba  = model.predict_proba(scaled)[0]

    st.success(f"### 🏆 Recommended Career: {role}")
    st.subheader("📈 Top 5 Career Matches")
    top5_idx   = proba.argsort()[-5:][::-1]
    top5_roles = target_encoder.inverse_transform(top5_idx)
    top5_probs = proba[top5_idx] * 100
    for i, (r, p) in enumerate(zip(top5_roles, top5_probs), 1):
        st.progress(int(p), text=f"{i}. {r}  —  {p:.1f}%")
