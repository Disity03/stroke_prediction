# Stroke Risk Prediction using Machine Learning

This project builds a machine learning model to predict the likelihood of a brain stroke based on various health and lifestyle parameters. It aims to assist in early detection and prevention by analyzing key risk factors such as age, hypertension, heart disease, BMI, smoking status, and more.

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

If using a virtual environment:

```bash
python3 -m venv ml-env
source ml-env/bin/activate
pip install -r requirements.txt
```

## How to Train

From the root directory:

  - Prepare data:
  
  ```bash
  python3 src/prepare_data.py
  ```

  - Train and Validate Model:
  
  ```bash
  python3 src/train_and_validate_model.py
  ```

  - Test Model:
  
  ```bash
  python3 src/test_model.py
  ```

This will:
- Preprocess and split the data
- Train your custom logistic regression
- Save the model weights, bias, and scaler
- Test your model

## Make a Prediction

To predict the probability of stroke for a new patient from terminal input:

```bash
python3 src/stroke_predictor.py
```

You'll be prompted to enter values for:
- Age, Hypertension, Heart disease, Glucose, BMI
- Gender, Marital status, Work type, Residence, Smoking status

## Dataset

Source: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 5,110 patient records
- Features include age, hypertension, heart disease, BMI, glucose, work type, etc.
- Highly imbalanced (fewer stroke-positive samples)

## Notes

- The threshold is set to 0.9 because negative examples are much more likely to have high predicted probabilities than positive examples are to have low ones
- In this dataset there was one example where gender was "Other", it was removed to improve training performance
