import pandas as pd
from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()

df=pd.read_csv('adult.csv',na_values=['?'])

#Filling missing values in workclass column with 'Unknown'
df['workclass']=df['workclass'].fillna('Unknown')

#Filling missing values in the occupation column with 'Unknown'
df['occupation']=df['occupation'].fillna('Unknown')

df=df.dropna(subset=['native.country'])

df=df.drop(columns=['fnlwgt'])

df.to_csv('adult_cleaned.csv',index=False)

print(df.describe())
print(df.info())

