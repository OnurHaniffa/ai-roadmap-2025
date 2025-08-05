# Day 13 – Model Comparison (Gradient Boosting & SVM)

## ✅ Goal:
Compare two new models — Gradient Boosting and Support Vector Machine — using full pipelines with preprocessing, and evaluate them using F1 (macro) scores via cross-validation.

---

## 🔧 Models Built:
1. **Gradient Boosting Classifier**
   - F1 Macro (Cross-Val): ~0.62
   - Test Set Accuracy: ~0.83
   - Test Set F1 (minority class): ~0.22
   - AUC Score: ~0.72

2. **Support Vector Machine (RBF Kernel)**
   - F1 Macro (Cross-Val): ~0.58

---

## 📊 Evaluation Summary:
- **Gradient Boosting outperformed SVM** in cross-validation and was selected for final testing.
- The model shows **very high accuracy**, but **low recall** on the minority class (Attrition = 1).
- **Class imbalance** clearly affects recall and F1 on minority class.

---

## 📦 Deliverables:
- `gb_model.pkl`: Saved trained pipeline
- `day13_model_comparison.ipynb`: Main notebook
- This `summary.md` file

---

## 🧠 Reflection:
- Stronger models like GB and SVM gave higher macro F1, but still struggle with imbalance.
- Important takeaway: **Cross-validation ≠ real-world performance** always.
- This sets the stage perfectly for **feature importance + fairness tuning** on Day 14.

