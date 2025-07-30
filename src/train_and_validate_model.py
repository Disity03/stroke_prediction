import pandas as pd
from logreg import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE

train_df = pd.read_csv("../data/stroke_train.csv")
train_df = train_df.dropna()
val_df = pd.read_csv("../data/stroke_val.csv")
val_df = val_df.dropna()

X_train = train_df.drop(columns=["stroke"])
y_train = train_df["stroke"]

X_val = val_df.drop(columns=["stroke"])
y_val = val_df["stroke"]

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)


model = LogisticRegression(learning_rate=0.05, n_iterations=1000000, lambda_ = 0.1, printnum = 20)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_val)
print("Validation report:")
print("Accuracy:", accuracy_score(y_val, y_pred)*100,"%")
print(classification_report(y_val, y_pred, zero_division=0))

joblib.dump(model, "../models/stroke_model.pkl")
