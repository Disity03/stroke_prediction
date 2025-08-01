import pandas as pd
from logreg import LogisticRegression
import joblib

# Choosing a model (Using pre-learned one or trying on new one)
while True:
	choice = input("Do you want to use pre-learned model or new one? (Old/New): ")
	if choice.lower() == "old":
		model = joblib.load("../models/stroke_model_save.pkl")
		break
	elif choice.lower() == "new":
		model = joblib.load("../models/stroke_model.pkl")
		break
	else:
		print("Wrong input!")


# Importing model and scaler
model = joblib.load("../models/stroke_model_save.pkl")
scaler = joblib.load("../models/stroke_scaler.pkl")


# Entering data
print("Enter new patient information:")

age = float(input("Age: "))

while True:
	hypertension = input("Hypertension (Yes/No): ")
	if hypertension.lower() == "yes" or hypertension.lower() == "no":
		break
	else:
		print("Wrong input!")

while True:
	heart_disease = input("Heart disease (Yes/No): ")
	if heart_disease.lower() == "yes" or heart_disease.lower() == "no":
		break
	else:
		print("Wrong input!")	

avg_glucose = float(input("Average glucose level: "))

bmi = float(input("BMI: "))

while True:
	gender = input("Gender (Male/Female): ").strip()
	if gender.lower() == "male" or gender.lower() == "female":
		break
	else:
		print("Wrong input!")	

while True:
	married = input("Ever married? (Yes/No): ").strip()
	if married.lower() == "yes" or married.lower() == "no":
		break
	else:
		print("Wrong input!")	

while True:
	work_type = input("Work type (Private/Self-employed/Govt_job/children/Never_worked): ").strip()
	if work_type.lower() == "private" or work_type.lower() == "self-employed" or work_type.lower() == "govt_job" or work_type.lower() == "children" or work_type.lower() == "never_worked":
		break
	else:
		print("Wrong input!")	

while True:
	residence = input("Residence type (Urban/Rural): ").strip()
	if residence.lower() == "urban" or residence.lower() == "rural":
		break
	else:
		print("Wrong input!")	

while True:
	smoking = input("Smoking status (Never smoked/Formerly smoked/Smokes): ").strip()
	if smoking.lower() == "never smoked" or smoking.lower() == "formerly smoked" or smoking.lower() == "smokes":
		break
	else:
		print("Wrong input!")	
		

# Formating data
raw_input = {
    "age": [age],
    "hypertension": [1 if hypertension.lower() == "yes" else 0],
    "heart_disease": [1 if heart_disease.lower() == "yes" else 0],
    "avg_glucose_level": [avg_glucose],
    "bmi": [bmi],
    "smoking_status": [
    	1.5 if smoking.lower() == "smokes" 
    	else 1 if smoking.lower() == "formerly smoked" 
    	else 0
    	],
    "gender_Male": [1 if gender.lower() == "male" else 0],
    "ever_married_Yes": [1 if married.lower() == "yes" else 0],
    "work_type_Never_worked": [1 if work_type.lower() == "never_worked" else 0],
    "work_type_Private": [1 if work_type.lower() == "private" else 0],
    "work_type_Self-employed": [1 if work_type.lower() == "self-employed" else 0],
    "work_type_children": [1 if work_type.lower() == "children" else 0],
    "Residence_type_Urban": [1 if residence.lower() == "urban" else 0],
}

input_df = pd.DataFrame(raw_input)

input_df[["age", "avg_glucose_level", "bmi"]] = scaler.transform(
    input_df[["age", "avg_glucose_level", "bmi"]]
)


# Predicting
prob = model.predict_proba(input_df.to_numpy(dtype=float))[0] * 100
print(f"Predicted danger of stroke: {prob:.4f}%")
