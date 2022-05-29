import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

gym=pd.read_excel(r'C:\Amey\Data Science\Gym app\dataGYM.xlsx')


gym=gym.drop('Unnamed: 6',axis=1)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded = label_encoder.fit(gym['Class'])
gym['Class'] = encoded.transform(gym['Class'])


X=gym.iloc[:,:3]
y=gym['Prediction']

#Sampling/Splitting of Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 1234)
from sklearn.ensemble import RandomForestClassifier #Non-Parametric Model

model_GYM = RandomForestClassifier(n_estimators=20)
model_GYM.fit(X_train, y_train)

expected = y_test
predicted = model_GYM.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

import pickle
pickle.dump(model_GYM, open("Model_GYM.pkl", "wb"))
model = pickle.load(open("Model_GYM.pkl", "rb"))
print(model.predict([[40,5.6,70]]))
