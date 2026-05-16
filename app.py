import streamlit as st
import joblib
import pandas as pd

model = joblib.load('student_model.joblib')

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title('🎓 Student Performance Predictor')
st.markdown('Predict a student final grade using Machine Learning.')
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader('📋 Student Profile')
    age = st.slider('Age', 15, 22, 17)
    studytime = st.selectbox('Weekly Study Time', [1,2,3,4],
        format_func=lambda x: {1:'<2 hours',2:'2-5 hours',3:'5-10 hours',4:'>10 hours'}[x])
    failures = st.selectbox('Past Failures', [0,1,2,3])

with col2:
    st.subheader('📈 Academic Record')
    absences = st.slider('Absences', 0, 30, 4)
    G1 = st.slider('First Period Grade G1', 0, 20, 10)
    G2 = st.slider('Second Period Grade G2', 0, 20, 12)

st.divider()

if st.button('Predict Final Grade', use_container_width=True):
    input_data = pd.DataFrame({
        'age': [age],
        'studytime': [studytime],
        'failures': [failures],
        'absences': [absences],
        'G1': [G1],
        'G2': [G2]
    })

    prediction = float(model.predict(input_data).item())
    prediction = max(0.0, min(20.0, prediction))
    percentage = round((prediction / 20) * 100, 1)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted Grade', str(round(prediction, 1)) + ' / 20')
    c2.metric('Percentage', str(percentage) + '%')

    if prediction >= 15:
        c3.metric('Status', 'Excellent')
        st.balloons()
        st.success('Outstanding performance!')
    elif prediction >= 10:
        c3.metric('Status', 'Pass')
        st.info('Satisfactory performance')
    else:
        c3.metric('Status', 'Fail')
        st.warning('Needs improvement')

    st.progress(prediction / 20)

st.divider()
st.caption('R2 Score: 0.863 | RMSE: 1.15 | Trained on Student Performance Dataset')