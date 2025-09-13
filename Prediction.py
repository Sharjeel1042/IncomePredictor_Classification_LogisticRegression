import joblib
import pandas as pd
import numpy as np

def predictingFromUserInput(userInput):
    model=joblib.load('Model.joblib')
    scaler=joblib.load('scaler.joblib')
    encoder=joblib.load('Encoder.joblib')

    numericalFeatures=joblib.load('numericalFeatures.joblib')
    categoricalFeatures=joblib.load('categoricalFeatures.joblib')

    user_df=pd.DataFrame([userInput])

    X_num=scaler.transform(user_df[numericalFeatures])
    X_cat=encoder.transform(user_df[categoricalFeatures])

    X_combine=np.hstack([X_num,X_cat])

    prediction=model.predict(X_combine)[0]
    prob=model.predict_proba(X_combine)[0]

    return prediction,prob

if __name__=='__main__':
    userInput={
        'age':int(input('Age: ')),
        'workclass':input('Work Class: '),
        'education':input('Education: '),
        'education.num':int(input('Education in number: ')),
        'marital.status':input('Marital Status: '),
        'occupation':input('Occupation: '),
        'relationship':input('Relationship: '),
        'race':input('Race: '),
        'sex':input('Sex: '),
        'capital.gain':int(input('Capital Gain: ')),
        'capital.loss':int(input('Capital Loss: ')),
        'hours.per.week':int(input('Hours per Week: ')),
        'native.country':input('Native Country: ')
    }

    prediction,probabilities=predictingFromUserInput(userInput)
    print('Prediction: ',prediction)
    print('Probabilities: ',probabilities)