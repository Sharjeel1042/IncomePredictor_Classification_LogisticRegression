import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import joblib
import numpy as np


dataFrame=pd.read_csv('adult_cleaned.csv')
X=dataFrame.drop(columns=['income'])
y=dataFrame['income']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

categoricalFeatures=['workclass','education','marital.status','occupation','relationship','race','sex','native.country']
numericalFeatures=['age','capital.gain','capital.loss','hours.per.week']

scaler=StandardScaler()
ohe=OneHotEncoder(handle_unknown='ignore', sparse_output=False)


#Scaling and encoding training data
X_train_scaled=scaler.fit_transform(X_train[numericalFeatures])
X_train_encoded=ohe.fit_transform(X_train[categoricalFeatures])


#Scaling and encoding test data
X_test_scaled=scaler.transform(X_test[numericalFeatures])
X_test_encoded=ohe.transform(X_test[categoricalFeatures])


#Combining data
X_train_trans=np.hstack([X_train_scaled,X_train_encoded])
X_test_trans=np.hstack([X_test_scaled,X_test_encoded])

#Model Training
model=LogisticRegression(max_iter=1000)
model.fit(X_train_trans,y_train)

y_pred=model.predict(X_test_trans)

#Saving scaling, encoding and model
joblib.dump(scaler,'scaler.joblib')
joblib.dump(ohe,'Encoder.joblib')
joblib.dump(model,'Model.joblib')
joblib.dump(numericalFeatures,'numericalFeatures.joblib')
joblib.dump(categoricalFeatures,'categoricalFeatures.joblib')

print("==========Performence Metrics==========")
print('Accuracy Score: ',accuracy_score(y_test,y_pred))
print('Classification Report: ',classification_report(y_test,y_pred))
