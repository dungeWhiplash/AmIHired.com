import streamlit as st
import pandas as pd
import joblib
import requests

url = 'https://github.com/dungeWhiplash/AmIHired/blob/main/model_pipeline.joblib'
response = requests.get(url)

model_path = 'model_pipeline.joblib'
with open(model_path, 'wb') as f:
    f.write(response.content)
loaded_model = joblib.load(model_path)

df = pd.read_csv("recruitment_data.csv")


st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Am I hired predictor 😅")

with st.form("my_form"):
    age=st.number_input(label='Age',min_value = 20, max_value = 50,step=1,format="%d")
    gender=st.number_input(label='Gender (Enter 0 for Male, 1 for Female)',min_value = 0, max_value = 1, step=1,format="%d")
    educationallevel=st.number_input(label='Highest Educational Level (Enter 1-BSc/BA, 2-BEng/BTech, 3-MS/MBA, 4-PhD)',min_value = 1, max_value = 4,step=1,format="%d")
    experienceyears=st.number_input(label='Experience in Years',min_value = 0, max_value = 15,step=1,format="%d")
    previouscompanies=st.number_input(label='Previous Companies',min_value = 1, max_value = 5,step=1,format="%d")
    distfromcom=st.number_input(label='Distance From Company in kilometers',min_value = 1.0, max_value = 50.0,step=0.01,format="%.2f")
    interviewscore=st.number_input(label='Interview Score',min_value = 0, max_value = 100,step=1,format="%d")
    skillscore=st.number_input(label='Skill Score',step=1,min_value = 0, max_value = 100,format="%d")
    personalityscore=st.number_input(label='Personality Score',step=1,min_value = 0, max_value = 100,format="%d")
    recruitmentstrat=st.number_input(label='Job application strategy (Enter 1: Aggresive, 2: Moderate, 3: Conservative)',min_value = 1, max_value = 3,step=1,format="%d")
    

    data=[[age, gender, educationallevel, experienceyears, previouscompanies, distfromcom, interviewscore, skillscore, personalityscore, recruitmentstrat]]

    submitted = st.form_submit_button("Submit")

if submitted:
    data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'EducationLevel': [educationallevel],
            'ExperienceYears': [experienceyears],
            'PreviousCompanies': [previouscompanies],
            'DistanceFromCompany': [distfromcom],
            'InterviewScore': [interviewscore],
            'SkillScore': [skillscore],
            'PersonalityScore': [personalityscore],
            'RecruitmentStrategy': [recruitmentstrat]
        })
    hire_prob = loaded_model.predict_proba(data)[:,1][0]
    decision=loaded_model.predict(data)
    st.write('Hiring decision:')
    if decision==1:
        st.success(f'High hiring chances. You have {hire_prob*100:.2f}% chances of getting hired.')
    elif decision==0:
        st.warning(f'Low Chances of getting hired with just {hire_prob*100:.2f}% probability')
