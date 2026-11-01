
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# 1. Load prepared datasets
print("Loading datasets...")

train = pd.read_csv("train_features.csv")
valid = pd.read_csv("valid_features.csv")

# 2. Separate features and target
TARGET = "clicked"

FEATURES = [
    "history_length",
    "candidate_in_history",
    "candidate_click_count",
    "time_since_last_click_hours",
    "time_bucket",
    "day_type",
]

X_train = train[FEATURES]
y_train = train[TARGET]

X_valid = valid[FEATURES]
y_valid = valid[TARGET]

# 3. Create LightGBM model
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
)

# 4. Train the model
print("Training LightGBM...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(30),
        lgb.log_evaluation(50),
    ],
)

# 5. Predict validation probabilities
predictions = model.predict_proba(X_valid)[:, 1]

# 6. Calculate ROC AUC
auc = roc_auc_score(y_valid, predictions)

print("\n==============================")
print(f"Validation ROC AUC: {auc:.6f}")
print("==============================")

# 7. Display feature importance
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance.to_string(index=False))

# 8. Save the model
model.booster_.save_model("lightgbm_baseline.txt")

print("\nModel saved: lightgbm_baseline.txt")
