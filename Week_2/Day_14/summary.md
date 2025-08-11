## Day 14 — Feature Importance Comparison (LR vs GB)

### Logistic Regression
- Strongest negative influence: JobRole_Research Director (-2.03), BusinessTravel_Non-Travel (-1.35)
- Strongest positive influence: JobRole_Sales Representative (+1.27)
- Overall, LR focuses heavily on categorical role and department features.

### Gradient Boosting
- Highest importance: TotalWorkingYears (0.14), Age (0.13)
- Focuses heavily on numeric tenure/age features, less on categorical one-hot variables.

### Observations
- LR and GB disagree on top predictors.
- LR’s directionality gives interpretable “push/pull” on attrition prediction.
- GB’s importance suggests tenure-related patterns dominate in its splits.
- This difference may contribute to GB’s poorer final F1 — over-reliance on fewer numeric features.

