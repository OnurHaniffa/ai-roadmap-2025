import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Week_2/Day_8/hr_data_ML_ready.csv')

X = df.drop('Attrition_num', axis=1)
y = df['Attrition_num']

x_test, x_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



