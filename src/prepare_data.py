import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# Importing and formatting data
df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")
df.drop(columns=["id"], inplace=True)
df.replace("Unknown", pd.NA, inplace=True)
df["age"] = df["age"].astype(int)
df["hypertension"] = df["hypertension"].astype(bool)
df["heart_disease"] = df["heart_disease"].astype(bool)
df["stroke"] = df["stroke"].astype(bool)
df["bmi"] = df["bmi"].astype(float)
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
df["smoking_status"] = df["smoking_status"].fillna(df["smoking_status"].mode()[0])
df = pd.get_dummies(df, columns=[
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
], drop_first=True)
scaler = StandardScaler()
df[["age", "avg_glucose_level", "bmi"]] = scaler.fit_transform(df[["age", "avg_glucose_level", "bmi"]])
joblib.dump(scaler, "../models/stroke_scaler.pkl")
X = df.drop(columns=["stroke"]) 
y = df["stroke"]  


# Spliting data into train, validation and test
X_train, X_temp, y_train, y_temp = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify = y
)

X_val, X_test, y_val, y_test = train_test_split(
	X_temp, y_temp, test_size=0.5, random_state=42 , stratify = y_temp
)


# Exporting data
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("../data/stroke_train.csv", index=False)
val_df.to_csv("../data/stroke_val.csv", index=False)
test_df.to_csv("../data/stroke_test.csv", index=False)

print("Data is ready!")
