# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

os.makedirs("models_saved", exist_ok=True)

# ----- EDIT THIS if your data filename is different -----
DATA_PATH =  "data/heart_disease_risk_dataset_earlymed.csv"
# If your CSV has a different target column name change TARGET_COL
TARGET_COL = "Heart_Risk"
# -------------------------------------------------------

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# Simple cleaning
df = df.dropna()
print("After dropna shape:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in data columns: {df.columns.tolist()}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# One-hot encode categoricals automatically
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipelines
pipelines = {
    "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
    "rf": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))]),
}

param_grid_rf = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 6, 10]
}

best_model = None
best_auc = 0
best_name = None

for name, pipe in pipelines.items():
    print("Training:", name)
    if name == "rf":
        grid = GridSearchCV(pipe, {"clf__n_estimators":[100,200],"clf__max_depth":[None,6,10]}, cv=4, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        pipe.fit(X_train, y_train)
        model = pipe

    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    print(f"{name} AUC: {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name

print("Best model:", best_name, "AUC:", best_auc)
joblib.dump(best_model, "models_saved/model.pkl")
joblib.dump(X.columns.tolist(), "models_saved/columns.pkl")
print("Saved model and columns to models_saved/")
