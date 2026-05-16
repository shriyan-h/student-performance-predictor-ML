import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('student_model.joblib')

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Header
st.title('🎓 Student Performance Predictor')
st.markdown('Predict a student\'s final grade using a Machine Learning model trained on real school data.')
st.divider()

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader('📋 Student Profile')
    age = st.slider('Age', 15, 22, 17)
    studytime = st.selectbox('Weekly Study Time',
        [1, 2, 3, 4],
        format_func=lambda x: {
            1:'<2 hours',
            2:'2-5 hours', 
            3:'5-10 hours',
            4:'>10 hours'
        }[x])
    failures = st.selectbox('Past Class Failures', [0, 1, 2, 3])

with col2:
    st.subheader('📈 Academic Record')
    absences = st.slider('Number of Absences', 0, 30, 4)
    G1 = st.slider('First Period Grade (G1)', 0, 20, 10)
    G2 = st.slider('Second Period Grade (G2)', 0, 20, 12)

st.divider()

# Predict
if st.button('🔮 Predict Final Grade', use_container_width=True):
    input_data = pd.DataFrame({
        'age': [age],
        'studytime': [studytime],
        'failures': [failures],
        'absences': [absences],
        'G1': [G1],
        'G2': [G2]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))

    st.divider()
    col3, col4, col5 = st.columns(3)

    with col3:
        grade_display = f'{float(prediction):.1f} / 20'
        st.metric('Predicted Grade', grade_display)
    with col4:
        percentage = (prediction / 20) * 100
        st.metric('Percentage', f'{float(percentage):.1f}%')
    with col5:
        if prediction >= 15:
            st.metric('Status', '🌟 Excellent')
        elif prediction >= 10:
            st.metric('Status', '👍 Pass')
        else:
            st.metric('Status', '❌ Fail')

    st.progress(float(prediction) / 20)

    if prediction >= 15:
        st.balloons()
        st.success('Outstanding performance predicted!')
    elif prediction >= 10:
        st.info('Satisfactory performance predicted')
    else:
        st.warning('This student may need additional support')

st.divider()
st.caption('Model trained on Student Performance Dataset | R2 Score: 0.863 | RMSE: 1.15')