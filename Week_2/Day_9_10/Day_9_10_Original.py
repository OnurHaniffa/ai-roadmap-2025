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


df = pd.read_csv('Week_2/Day_8/hr_data_ML_ready.csv')
X = df.drop(columns=['Attrition_num'])
y = df['Attrition_num']


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------------------------------------------------------------------
# KNN: Tune k and evaluate
# ------------------------------------------------------------------------------------------------------------------------------
k_values = range(1, 21)
knn_acc, knn_prec, knn_rec, knn_f1 = [], [], [], []

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_scaled, y_train)
    y_pred_knn = knn.predict(x_test_scaled)
    knn_acc.append(accuracy_score(y_test, y_pred_knn))
    knn_prec.append(precision_score(y_test, y_pred_knn))
    knn_rec.append(recall_score(y_test, y_pred_knn))
    knn_f1.append(f1_score(y_test, y_pred_knn))

best_k = k_values[np.argmax(knn_f1)]
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train_scaled, y_train)
y_pred_knn = knn.predict(x_test_scaled)

acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# ------------------------------------------------------------------------------------------------------------------------------
# Decision Tree: Tune max_depth (2–10) and plot
# ------------------------------------------------------------------------------------------------------------------------------
depth_values = range(2, 11)
acc_tree, prec_tree, rec_tree, f1_tree = [], [], [], []

for depth in depth_values:
    tree = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
    tree.fit(x_train, y_train)
    y_pred_tree = tree.predict(x_test)
    acc_tree.append(accuracy_score(y_test, y_pred_tree))
    prec_tree.append(precision_score(y_test, y_pred_tree))
    rec_tree.append(recall_score(y_test, y_pred_tree))
    f1_tree.append(f1_score(y_test, y_pred_tree))

best_depth = depth_values[np.argmax(f1_tree)]
tree = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=42)
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)

acc_tree_final = accuracy_score(y_test, y_pred_tree)
prec_tree_final = precision_score(y_test, y_pred_tree)
rec_tree_final = recall_score(y_test, y_pred_tree)
f1_tree_final = f1_score(y_test, y_pred_tree)

# Plot metrics vs depth
tree_metrics = {"Accuracy": acc_tree, "Precision": prec_tree, "Recalls": rec_tree, "F1": f1_tree}
for metric_name, values in tree_metrics.items():
    plt.figure(figsize=(8,5))
    plt.plot(depth_values, values, marker='o')
    plt.title(f"Decision Tree {metric_name} vs. Depth")
    plt.xlabel("Max Depth")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression (scaled, no SMOTE)
# ------------------------------------------------------------------------------------------------------------------------------
log_reg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
log_reg.fit(x_train_scaled, y_train)
y_pred_log = log_reg.predict(x_test_scaled)

acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

# With SMOTE (on scaled x_train only)
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

log_reg_smote = LogisticRegression(random_state=42, max_iter=1000)
log_reg_smote.fit(x_train_balanced, y_train_balanced)
y_pred_log_smote = log_reg_smote.predict(x_test_scaled)

acc_log_smote = accuracy_score(y_test, y_pred_log_smote)
prec_log_smote = precision_score(y_test, y_pred_log_smote)
rec_log_smote = recall_score(y_test, y_pred_log_smote)
f1_log_smote = f1_score(y_test, y_pred_log_smote)

# ------------------------------------------------------------------------------------------------------------------------------
# Random Forest (original data, no scaling)
# ------------------------------------------------------------------------------------------------------------------------------
forest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
forest.fit(x_train, y_train)
y_pred_forest = forest.predict(x_test)

acc_forest = accuracy_score(y_test, y_pred_forest)
prec_forest = precision_score(y_test, y_pred_forest)
rec_forest = recall_score(y_test, y_pred_forest)
f1_forest = f1_score(y_test, y_pred_forest)

# ------------------------------------------------------------------------------------------------------------------------------
# Random Forest Hyperparameter Tuning
# ------------------------------------------------------------------------------------------------------------------------------
depth_values = [2, 4, 6, 8, 10]
split_values = [2, 5, 10]
weights = [None, 'balanced']

print("\nRandom Forest Hyperparameter Tuning Results:")
print(f"{'Depth':<5} {'Split':<6} {'Weight':<10} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1':<8}")
for depth in depth_values:
    for split in split_values:
        for weight in weights:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=depth,
                min_samples_split=split,
                class_weight=weight,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            print(f"{depth:<5} {split:<6} {str(weight):<10} {acc:<9.3f} {prec:<10.3f} {rec:<8.3f} {f1:<8.3f}")

# -----------------------
# Final Model Comparison Table
# -----------------------
comparison = pd.DataFrame({
    "Model": [
        f"KNN (best k={best_k})",
        f"Decision Tree (depth={best_depth})",
        "Logistic Regression (original)",
        "Logistic Regression (SMOTE)",
        "Random Forest (default)"
    ],
    "Accuracy": [acc_knn, acc_tree_final, acc_log, acc_log_smote, acc_forest],
    "Precision": [prec_knn, prec_tree_final, prec_log, prec_log_smote, prec_forest],
    "Recall": [rec_knn, rec_tree_final, rec_log, rec_log_smote, rec_forest],
    "F1 Score": [f1_knn, f1_tree_final, f1_log, f1_log_smote, f1_forest]
})

print("\nModel Comparison (All Models):")
print(comparison)
