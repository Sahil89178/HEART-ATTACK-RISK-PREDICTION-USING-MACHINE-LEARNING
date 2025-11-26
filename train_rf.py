import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("processed_heart.csv")

X = df.drop("HeartDisease", axis=1).values
y = df["HeartDisease"].values

scaler = joblib.load("models/scaler.pkl")
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=400, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(rf, "models/rf_model.pkl")
