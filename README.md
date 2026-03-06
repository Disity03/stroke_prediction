# Stroke Risk Prediction Using Machine Learning

This project builds a machine learning model to predict the probability of stroke based on various health and lifestyle parameters. Its goal is to support early detection and prevention by analyzing key risk factors such as age, hypertension, heart disease, BMI, smoking status, and others.

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

If you use a virtual environment:

```bash
python3 -m venv ml-env
source ml-env/bin/activate
pip install -r requirements.txt
```

## How to Train

From the **src** directory:

- Prepare the data:
  
  ```bash
  python3 prepare_data.py
  ```

- Train and validate the model:
  
  ```bash
  python3 train_and_validate_model.py
  ```

- Test the model:
  
  ```bash
  python3 test_model.py
  ```

This will:
- Preprocess and split the data
- Train a custom logistic regression model
- Save model parameters, bias, and scaler
- Test your model

## Making a Prediction

To predict stroke probability for a new patient from the terminal:

```bash
python3 stroke_predictor.py
```

You will be prompted to enter values for:
- Age, hypertension, heart disease, glucose, BMI
- Gender, marital status, work type, residence type, smoking status

## Dataset

Source: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
- 5,110 patient records  
- Features include age, hypertension, heart disease, BMI, glucose, work type, etc.  
- Highly imbalanced (few positive stroke samples)

## Notes

- The threshold is set to 0.97 because negative samples are much more likely to have high predicted probabilities than positive samples are to have low ones  
- In this dataset, there was one sample where gender was "Other", and it was removed to improve training performance  

## Comparisons

On the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) page, there are also solutions from other users, where the decision threshold is mostly set to 0.5 (default). For comparison, here are my results with that threshold:

```bash
Accuracy: 77.1037%

              precision    recall  f1-score   support

       False       0.98      0.77      0.87       486
        True       0.14      0.72      0.24        25

    accuracy                           0.77       511
   macro avg       0.56      0.75      0.55       511
weighted avg       0.94      0.77      0.83       511
```

- [Marwan ElMahalawy](https://www.kaggle.com/code/marwanelmahalawy/stroke-logistic-regression)

Logistic regression was also used here, with different data processing, and the results are as follows:

```bash
Accuracy: 74.8472%

              precision    recall  f1-score   support

       False       0.98      0.75      0.85       929
        True       0.15      0.79      0.25        53

    accuracy                           0.75       982
   macro avg       0.57      0.77      0.55       982
weighted avg       0.94      0.75      0.82       982
```

- [Josh](https://www.kaggle.com/code/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5#What-about-Logistic-Regression?)

This is the top-rated solution on Kaggle. [Josh](https://www.kaggle.com/code/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5) used other techniques as well, but for comparison I will include logistic regression:

```bash
Accuracy: 75.8177%
              precision    recall  f1-score   support

           0       0.97      0.77      0.86      3404
           1       0.11      0.60      0.19       173

    accuracy                           0.76      3577
   macro avg       0.54      0.68      0.53      3577
weighted avg       0.93      0.76      0.83      3577
```

He concluded that logistic regression worked best overall, while also testing Random Forest and Support Vector Machine.

- [Nima Pourmoradi](https://www.kaggle.com/code/nimapourmoradi/healthcare-stroke)

```bash
Accuracy: 69.8531%
              precision    recall  f1-score   support

           0       0.99      0.59      0.74      1179
           1       0.08      0.84      0.14        49

    accuracy                           0.60      1228
   macro avg       0.53      0.71      0.44      1228
weighted avg       0.95      0.60      0.71      1228
```

## Conclusion

Although the results are not perfect, logistic regression turns out to be one of the best models for this problem. Results also depend on what probability threshold we consider sufficient for stroke occurrence. Overall, if we do not treat this strictly as a binary 0/1 problem, logistic regression appears capable of providing a sufficiently good estimate of stroke risk, while the physician remains responsible for deciding how to act on that result.
