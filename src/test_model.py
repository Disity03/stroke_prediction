import pandas as pd
from logreg import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

test_df = pd.read_csv("../data/stroke_test.csv")
test_df = test_df.dropna()

X_test = test_df.drop(columns=["stroke"])
y_test = test_df["stroke"]

model = joblib.load("../models/stroke_model.pkl")

y_pred = model.predict(X_test)
print("Test report:")
print("Accuracy:", accuracy_score(y_test, y_pred)*100,"%")
print(classification_report(y_test, y_pred, zero_division=0))
