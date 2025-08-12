# Day 9–10: KNN, Decision Tree, Logistic Regression, and Random Forest with Tuning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def load_and_prepare_data():
    df = pd.read_csv('Week_2/Day_8/hr_data_ML_ready.csv')
    x = df.drop(columns=['Attrition_num'])
    y = df['Attrition_num']


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    return x_train,x_test,y_train,y_test

# ------------------------------------------------------------------------------------------------------------------------------
# KNN: Tune k and evaluate
# ------------------------------------------------------------------------------------------------------------------------------

def train_knn(x_train ,x_test, y_train ,y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    k_values = range(1, 21)
    knn_acc, knn_prec, knn_rec, knn_f1 = [], [], [], []




    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        knn_f1.append(f1_score(y_test, y_pred))

    best_k = k_values[np.argmax(knn_f1)]
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(x_train_scaled, y_train)
    return model,scaler


# ------------------------------------------------------------------------------------------------------------------------------
# Decision Tree: Tune max_depth (2–10) and plot
# ------------------------------------------------------------------------------------------------------------------------------

def train_decision_tree(x_train,x_test,y_train,y_test):

    depth_values = range(2, 11)
    f1_tree = []

    for depth in depth_values:
        model= DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        f1_tree.append(f1_score(y_test, y_pred))

    best_depth = depth_values[np.argmax(f1_tree)]
    model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=42)
    model.fit(x_train, y_train)
    return model
#-------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression (scaled, no SMOTE)
# ------------------------------------------------------------------------------------------------------------------------------
def train_log_reg(x_train,x_test,y_train):
    scaler= StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(x_train_scaled, y_train)
    
    smote=SMOTE(random_state=42)
    x_train_balanced,y_train_balanced=smote.fit_resample(x_train_scaled,y_train)
    model_smote= LogisticRegression(random_state=42,max_iter=1000)
    model_smote.fit(x_train_scaled,y_train)

    return model,model_smote,scaler

# ------------------------------------------------------------------------------------------------------------------------------
# Random Forest (original data, no scaling)
# ------------------------------------------------------------------------------------------------------------------------------
def train_random_forest(x_train,y_train):
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model

def get_trained_models():
    x_train,x_test,y_train,y_test= load_and_prepare_data()

    knn,knn_scaler= train_knn(x_train,x_test,y_train,y_test)
    decision_tree= train_decision_tree(x_train,x_test,y_train,y_test)
    log_reg,log_reg_smote,log_scaler=train_log_reg(x_train,x_test,y_train)
    random_forest= train_random_forest(x_train,y_train)

    models = {
        "knn": knn,
        "decision_tree": decision_tree,
        "logistic": log_reg,
        "logistic_smote": log_reg_smote,
        "random_forest": random_forest
    }

    scalers = {
        "knn": knn_scaler,
        "logistic": log_scaler
    }

    return models, scalers, x_train, x_test, y_train, y_test


# ----------------------------------------------------------
# Debug mode 
# ----------------------------------------------------------
if __name__ == "__main__":
    models, scalers, x_train, x_test, y_train, y_test = get_trained_models()
    print("Trained models:", list(models.keys()))





