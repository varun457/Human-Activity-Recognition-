import os, json, random
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

# ─── Paths ────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "human+activity+recognition+using+smartphones",
                         "UCI HAR Dataset", "UCI HAR Dataset")

# ─── Load artifacts once at startup ───────────────────────
model   = joblib.load(os.path.join(BASE, "model.pkl"))
acc     = json.load(open(os.path.join(BASE, "accuracy.json")))["accuracy"]
report  = json.load(open(os.path.join(BASE, "classification_report.json")))

# Load test data for random-sample predictions
X_test = pd.read_csv(os.path.join(DATA_ROOT, "test", "X_test.txt"),
                     sep=r'\s+', header=None)
y_test = pd.read_csv(os.path.join(DATA_ROOT, "test", "y_test.txt"),
                     header=None).squeeze()

# Load feature names
features_df = pd.read_csv(os.path.join(DATA_ROOT, "features.txt"),
                          sep=r'\s+', header=None, names=['idx', 'name'])
feature_names = features_df['name'].tolist()
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
X_test.columns = feature_names

activity_map = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
}
y_test = y_test.map(activity_map)

# Activity icons / colors for the UI  (emoji + CSS class)
ACTIVITY_META = {
    'WALKING':            {'icon': '🚶', 'color': '#4ade80'},
    'WALKING_UPSTAIRS':   {'icon': '⬆️', 'color': '#60a5fa'},
    'WALKING_DOWNSTAIRS': {'icon': '⬇️', 'color': '#f472b6'},
    'SITTING':            {'icon': '🪑', 'color': '#facc15'},
    'STANDING':           {'icon': '🧍', 'color': '#a78bfa'},
    'LAYING':             {'icon': '🛌', 'color': '#fb923c'},
}

# ─── Routes ───────────────────────────────────────────────

@app.route("/")
def index():
    # Prepare per-class metrics from classification report
    class_labels = list(activity_map.values())
    class_metrics = []
    for label in class_labels:
        m = report.get(label, {})
        class_metrics.append({
            'label':     label,
            'precision': round(m.get('precision', 0) * 100, 1),
            'recall':    round(m.get('recall', 0) * 100, 1),
            'f1':        round(m.get('f1-score', 0) * 100, 1),
            'support':   int(m.get('support', 0)),
            'icon':      ACTIVITY_META[label]['icon'],
            'color':     ACTIVITY_META[label]['color'],
        })

    return render_template("index.html",
                           accuracy=round(acc * 100, 2),
                           class_metrics=class_metrics,
                           total_train=7352,
                           total_test=2947,
                           n_features=561,
                           n_estimators=200)


@app.route("/predict", methods=["POST"])
def predict():
    """Pick a random test sample, predict, and return JSON."""
    idx = random.randint(0, len(X_test) - 1)
    sample = X_test.iloc[idx:idx+1]
    predicted = model.predict(sample)[0]
    actual    = y_test.iloc[idx]
    meta      = ACTIVITY_META.get(predicted, {'icon': '❓', 'color': '#888'})
    return jsonify({
        'index':     int(idx),
        'predicted': predicted,
        'actual':    actual,
        'correct':   predicted == actual,
        'icon':      meta['icon'],
        'color':     meta['color'],
    })


@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve root-level images (confusion_matrix.png, feature_importance.png)."""
    return send_from_directory(BASE, filename)


if __name__ == "__main__":
    print("🚀  HAR Dashboard running at  http://localhost:5000")
    app.run(debug=True, port=5000)
