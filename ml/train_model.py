import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
files = ["MOVE.csv", "LEFT_CLICK.csv", "RIGHT_CLICK.csv", "DRAG.csv"]
data = []

for file in files:
    df = pd.read_csv(file, header=None)
    data.append(df)

data = pd.concat(data, ignore_index=True)

X = data.iloc[:, :-1]   # features
y = data.iloc[:, -1]    # labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "gesture_model.pkl")

print("Model trained and saved as gesture_model.pkl")
