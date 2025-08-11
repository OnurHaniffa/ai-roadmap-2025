# Elite AI Engineer Roadmap (18–24 Months)

## Current Status
- **Date:** [Fill in Today]
- **Phase:** Phase 0 – Foundation (Week 3 starting tomorrow)
- **Day Count:** 14 days complete
- **Best Model (HR Dataset):** Logistic Regression pipeline (SMOTE + tuned threshold)
- **Skills Covered:**
  - Python (intermediate+), Pandas, NumPy, Seaborn, Matplotlib
  - Data cleaning, EDA, preprocessing
  - ML models: LR, DT, RF, KNN, GB, SVM
  - Evaluation: Accuracy, Precision/Recall, F1 (macro & weighted), ROC AUC, PR AUC
  - Imbalance handling: SMOTE, undersampling, class weights
  - Pipelines with ColumnTransformer
  - Feature importance: coefficients, tree importances
  - Model saving/loading (`joblib`)

---

## Long-Term Plan

### Phase 0 – Foundation (Weeks 1–8)
*Already mid-way here (starting Week 3)*  
- Apply all fundamentals to multiple datasets
- Deploy first small ML API
- Learn Docker basics
- First cloud deployment (free tier)

### Phase 1 – Real-World ML & Deployment (Months 3–5)
- FastAPI & Docker deployments
- Streamlit dashboards
- Cloud hosting (AWS/GCP/Azure basics)
- Database integration
- Project portfolio apps

### Phase 2 – Data Engineering Skills (Months 6–7)
- SQL (advanced queries, joins, windows)
- Data pipelines (Airflow/Prefect)
- Cloud storage & ETL pipelines

### Phase 3 – Deep Learning Foundations (Months 8–9)
- Neural networks (PyTorch/TensorFlow)
- MLPs for tabular data
- GPU training

### Phase 4 – Computer Vision Mastery (Months 10–12)
- Image preprocessing
- CNNs, transfer learning, augmentation
- Object detection, video analysis

### Phase 5 – NLP & LLM Mastery (Months 13–15)
- Text preprocessing, embeddings
- Transformers (Hugging Face)
- Fine-tuning GPT/BERT/LLaMA
- RAG pipelines

### Phase 6 – Advanced AI Engineering & MLOps (Months 16–18)
- CI/CD, model monitoring
- Drift detection, retraining pipelines
- Model versioning (MLflow)
- Scaling with Kubernetes

### Phase 7 – Specialization & Capstone (Months 19–24)
- Large biomedical AI project
- Fully deployed cloud system
- Public portfolio + blog posts

---

## Daily Log

### Day 14 (Today)
- Compared LR vs GB feature importance
- Learned pipeline internals (ColumnTransformer, mini-pipeline vs direct transformer)
- Extracted transformed feature names
- Reviewed CV vs true test set F1 differences
- Merged LR & GB importance tables

**Next Planned Task (Day 15 – Start of Week 3):**
- Pick a **new dataset** for real-world project phase
- Full EDA + preprocessing pipeline
- Baseline model + quick tuning
- Prepare for deployment steps (FastAPI) starting Day 17
