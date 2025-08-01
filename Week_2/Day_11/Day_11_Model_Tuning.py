# Day 11: Model Evaluation & Hyperparameter Tuning

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import sys, os
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import trained models + data from Day 9–10
from Day_9_10.Day_9_10_ML_Models_Refactored import get_trained_models

# Load everything
models, scalers, x_train, x_test, y_train, y_test = get_trained_models()


results=[]

for name,model in models.items():
    if name in scalers:
        scaler=scalers[name]
        x_test_scaled=scaler.transform(x_test)
        y_pred=model.predict(x_test_scaled)
        y_proba=model.predict_proba(x_test_scaled)[:,1]
    else:
        y_pred=model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else None


    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_pred,y_test,zero_division=0)

    try:
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            conf_matrix=confusion_matrix(y_test,y_pred)
            print(f'Confusion matrix for {name}:\n{conf_matrix}')
        else:
            roc_auc = None
    except:
        roc_auc = None

   


    results.append({'Model':name,
                    'Accuracy':accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1':f1,
                    'Roc_Auc':roc_auc })

results_df=pd.DataFrame(results)
print(results_df)
print(results_df.sort_values(by='F1',ascending=False).reset_index(drop=True)),


for name,model in models.items():
    if name in scalers:
        x_test_scaled=scalers[name].transform(x_test)
        x_input=x_test_scaled
    else:
        x_input=x_test

    y_pred=model.predict(x_input)

    if hasattr(model,'predict_proba'):
        y_proba=model.predict_proba(x_input)[:,1]
        # RocCurveDisplay.from_predictions(y_test,y_proba,name=name).plot(ax=plt.gca())

        # plt.plot([0,1],[0,1],'k--',label='Random Guess')
        # plt.legend()
        # plt.grid(True)
        # plt.title(f'ROC Curve Comparison')
        # plt.show()
    
    # cm=confusion_matrix(y_test,y_pred)
    # sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    # plt.xlabel('Y predictions')
    # plt.ylabel('Actual Value')
    # plt.title(f'Confusion Matrix for {name}')

    #---------------------------------------------------------------------------------------------------------------

param_grid = {
    'penalty': ['l2'],  # L2 regularization (ridge)
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # From strong regularization to weak
    'solver': ['lbfgs', 'liblinear'],  # Compatible solvers for L2
    'max_iter': [100, 500, 1000]
}

scaler=scalers['logistic']
x_test_scaled=scaler.transform(x_test)
x_train_scaled=scaler.transform(x_train)


    
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
grid=GridSearchCV(log_reg,param_grid,cv=5,scoring='f1',n_jobs=-1,verbose=1)
grid.fit(x_train_scaled,y_train)

   
print(f'Best parameter : {grid.best_params_}'
      f'Best f1 score: {grid.best_score_}')

best_model=grid.best_estimator_
y_pred=best_model.predict(x_test_scaled)

print("Confusion matrix for the best log reg model:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report for the best log reg model:")
print(classification_report(y_test, y_pred))

import joblib


joblib.dump(best_model, 'best_logistic_model.joblib')




        