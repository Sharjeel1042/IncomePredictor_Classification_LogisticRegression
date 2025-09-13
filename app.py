import streamlit as st
import pandas as pd
import joblib
from Prediction import predictingFromUserInput


model=joblib.load('Model.joblib')
st.title("Machine Learning Model for Adult Census Data to predict user Income ")

#Age
age=st.slider("Age",17,90,30)
#WorkClass
workclass=st.selectbox("Work Class",['Federal-gov','Local-gov','Never-worked','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Unknown','Without-pay'])
#Education
education=st.selectbox("Education",['Preschool',"1st-4th",'5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate'])
#MaritalStatus
maritalStatus=st.selectbox("Marital Status",['Divorced','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Never-married','Separated','Widowed'])
#Occupation
occupation=st.selectbox("Occupation",['Adam-clerical','Armed-Forces','Craft-repair','Exec-managerial','farming-Fishing','Handlers-cleaners','Machine-op-inspct','Priv-house-serv','Prof-specialty','Protective-serv','Sales','Tech-support','transport-moving','Other-service','Unknown'])
#Relationship
relationship=st.selectbox("Relationship",['Husband','Not-in-family','Other-relative','Own-child','Unmarried','Wife'])
#Race
race=st.radio("Race",['White','Black'])
#Sex
sex=st.radio("Gender",['Male','Female'])
#CapitalGain
capitalGain=st.slider("Capital Gain",0,100000,1000,step=500)
#CapitalLoss
capitalLoss = st.slider("Capital Loss", 0,500,100,step=10)
#HoursPerWeek
hoursPerWeek=st.slider("Hours Per Week",1,100,2)

data=pd.read_csv('adult_cleaned.csv')
countries=data['native.country'].dropna().unique()
countries=sorted(countries)
nativeCountry=st.selectbox("Native Country",countries)

#EducationNum
educationNum=0

if education== 'Preschool':
    educationNum=1
elif education=='1st-4th':
    educationNum=2
elif education=='5th-6th':
    educationNum=3
elif education=='7th-8th':
    educationNum=4
elif education=='9th':
    educationNum=5
elif education=='10th':
    educationNum=6
elif education=='11th':
    educationNum=7
elif education=='12th':
    educationNum=8
elif education=='HS-grad':
    educationNum=9
elif education=='Some-college':
    educationNum=10
elif education=='Assoc-voc':
    educationNum=11
elif education=='Assoc-acdm':
    educationNum=12
elif education=='Bachelors':
    educationNum=13
elif education=='Masters':
    educationNum=14
elif education=='Prof-school':
    educationNum=15
elif education=='Doctorate':
    educationNum=16

user_input={
    'age':age,
    'workclass':workclass,
    'education':education,
    'education.num':educationNum,
    'marital.status':maritalStatus,
    'occupation':occupation,
    'relationship':relationship,
    'race':race,
    'sex':sex,
    'capital.gain':capitalGain,
    'capital.loss':capitalLoss,
    'hours.per.week':hoursPerWeek,
    'native.country':nativeCountry
}

# Adding predict button
if st.button("Predict Income"):
    prediction, probability = predictingFromUserInput(user_input)

    # Displaying results
    income_class = "≤50K" if prediction == 0 else ">50K"
    st.success(f"Predicted Income: {income_class}")



