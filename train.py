import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

# Fraud column = Class (1 fraud, 0 normal)
X = df.drop("Class", axis=1)
y = df["Class"]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print("Testing model...")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Saving model...")

joblib.dump(model, "model.pkl")

print("✅ Model saved as model.pkl")

