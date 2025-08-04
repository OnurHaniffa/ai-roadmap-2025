# Day 12 Summary: ML Pipeline and Model Tuning

## ✅ What I Did
- Loaded cleaned HR dataset (categoricals still unencoded)
- Built a full ML pipeline with:
  - ColumnTransformer (OneHotEncoder + StandardScaler)
  - LogisticRegression classifier
- Used GridSearchCV (5-fold CV) to tune:
  - C: [0.1, 1, 10]
  - penalty: ['l2']
  - solver: ['lbfgs', 'liblinear']
- Found best parameters: `{'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}`
- Best cross-val F1: **~0.42**
- Test F1 score: **~0.32**, Test accuracy: **~0.84**

## 💡 What I Learned
- Pipelines automate preprocessing + modeling
- Difference between cross-val score and test score
- What `.pkl` files and `joblib` do
- Importance of feature consistency when reusing a pipeline

## 🧠 Next Steps
- Consider SMOTE to improve recall on minority class (Day 13?)
- Try other classifiers (RandomForest, XGBoost)
- Use SHAP/feature importance to understand the model

