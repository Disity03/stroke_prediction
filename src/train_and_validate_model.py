import pandas as pd
from logreg import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE
import datetime

# Importing data
train_df = pd.read_csv("../data/stroke_train.csv")
train_df = train_df.dropna()
val_df = pd.read_csv("../data/stroke_val.csv")
val_df = val_df.dropna()

X_train = train_df.drop(columns=["stroke"])
y_train = train_df["stroke"]

X_val = val_df.drop(columns=["stroke"])
y_val = val_df["stroke"]


# Resampling data because data is oversampled
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)


# Training of a model
model = LogisticRegression(learning_rate=0.04, n_iterations=500000, lambda_ = 0.1, printnum = 50)
iter_report = model.fit(X_resampled, y_resampled)


# Validation
y_pred = model.predict(X_val)
s = "Validation report:\n"
s+= f"Accuracy:{(accuracy_score(y_val, y_pred)*100): .4f}%\n"
s+= classification_report(y_val, y_pred, zero_division=0)
print(s)


# Writing log
filename = f"../outputs/train_and_validation_report_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
f = open(filename,"a")
f.write(iter_report)
f.write(s)


# Exporting model
joblib.dump(model, "../models/stroke_model.pkl")
