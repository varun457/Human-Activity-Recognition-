# ─── 1. IMPORTS ───────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')       # non-interactive backend (no plt.show popups)
import matplotlib.pyplot as plt
import seaborn as sns
import json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─── 2. LOAD DATA ─────────────────────────────────────────
DATA_ROOT = "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"

X_train = pd.read_csv(f"{DATA_ROOT}/train/X_train.txt", sep=r'\s+', header=None)
y_train = pd.read_csv(f"{DATA_ROOT}/train/y_train.txt", header=None).squeeze()
X_test  = pd.read_csv(f"{DATA_ROOT}/test/X_test.txt", sep=r'\s+', header=None)
y_test  = pd.read_csv(f"{DATA_ROOT}/test/y_test.txt", header=None).squeeze()

# Load real feature names from features.txt
features_df = pd.read_csv(f"{DATA_ROOT}/features.txt", sep=r'\s+', header=None, names=['idx', 'name'])
feature_names = features_df['name'].tolist()

# Handle duplicate feature names by appending a suffix
seen = {}
unique_names = []
for name in feature_names:
    if name in seen:
        seen[name] += 1
        unique_names.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 0
        unique_names.append(name)
feature_names = unique_names

X_train.columns = feature_names
X_test.columns  = feature_names

print("Training samples:", X_train.shape)  # (7352, 561)
print("Testing samples:",  X_test.shape)   # (2947, 561)

# ─── 3. ACTIVITY MAPPING ──────────────────────────────────
activity_map = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

y_train = y_train.map(activity_map)
y_test  = y_test.map(activity_map)

# ─── 4. TRAIN THE MODEL ───────────────────────────────────
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
rfc.fit(X_train, y_train)

# ─── 5. PREDICT & EVALUATE ────────────────────────────────
y_pred = rfc.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy Score:", acc)

report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ─── 6. SAVE MODEL & RESULTS ─────────────────────────────
joblib.dump(rfc, "model.pkl")
print("Model saved to model.pkl")

json.dump({"accuracy": acc}, open("accuracy.json", "w"))
json.dump(report, open("classification_report.json", "w"), indent=2)
print("Results saved to accuracy.json & classification_report.json")

# ─── 7. SAMPLE PREDICTION ─────────────────────────────────
print("Sample prediction:", rfc.predict(X_test.iloc[0:1]))

# ─── 8. FEATURE IMPORTANCE (with real feature names) ──────
importances = pd.Series(rfc.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)[:10]

print("Top 10 feature importances:\n", importances)

plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")

# ─── 9. CONFUSION MATRIX ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
labels = list(activity_map.values())

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

print("\n✅ Done! Run  python app.py  to launch the web dashboard.")