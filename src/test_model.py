import pandas as pd
from logreg import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import datetime

test_df = pd.read_csv("../data/stroke_test.csv")
test_df = test_df.dropna()

X_test = test_df.drop(columns=["stroke"])
y_test = test_df["stroke"]

model = joblib.load("../models/stroke_model.pkl")



y_pred = model.predict(X_test)
s = "Test report:\n"
s+= f"Accuracy:{(accuracy_score(y_test, y_pred)*100): .4f}%\n"
s+= classification_report(y_test, y_pred, zero_division=0)
print(s)

filename = f"../outputs/test_report_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
f = open(filename,"a")
f.write(s)
