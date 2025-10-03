# Predikcija rizika od moždanog udara pomoću mašinskog učenja

Ovaj projekat gradi model mašinskog učenja za predviđanje verovatnoće moždanog udara na osnovu različitih zdravstvenih i životnih parametara. Cilj mu je da pomogne u ranom otkrivanju i prevenciji analizom ključnih faktora rizika kao što su starost, hipertenzija, srčane bolesti, BMI, status pušenja i drugi.

## Zahtevi

Instalacija zavisnosti:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

Ako koristite virtuelno okruženje:

```bash
python3 -m venv ml-env
source ml-env/bin/activate
pip install -r requirements.txt
```

## Kako trenirati

Iz **src** direktorijuma:

- Pripremite podatke:
  
  ```bash
  python3 prepare_data.py
  ```

- Trenira i validira model:
  
  ```bash
  python3 train_and_validate_model.py
  ```

- Testira model:
  
  ```bash
  python3 test_model.py
  ```

Ovo će:
- Predobraditi i podeliti podatke
- Trenirati prilagođenu logističku regresiju
- Sačuvati parametre modela, bias i scaler
- Testirati vaš model

## Pravljenje predikcije

Da biste predvideli verovatnoću moždanog udara za novog pacijenta iz terminala:

```bash
python3 stroke_predictor.py
```

Od vas će biti zatraženo da unesete vrednosti za:
- Godine, hipertenziju, srčane bolesti, glukozu, BMI
- Pol, bračni status, tip posla, mesto stanovanja, status pušenja

## Skup podataka

Izvor: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
- 5.110 zapisa o pacijentima  
- Karakteristike uključuju starost, hipertenziju, srčane bolesti, BMI, glukozu, tip posla itd.  
- Veoma neuravnotežen (malo uzoraka sa pozitivnim moždanim udarom)

## Napomene

- Prag je podešen na 0.97 jer je mnogo verovatnije da negativni primeri imaju visoke predviđene verovatnoće nego da pozitivni primeri imaju niske  
- U ovom skupu podataka postojao je jedan primer gde je pol bio „Other“, koji je uklonjen da bi se poboljšale performanse treninga  

## Poredjenja

Na sajtu [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) postoje i rešenja drugih, pa su ovde prikazana poređenja:

- [Marwan ElMahalawy](https://www.kaggle.com/code/marwanelmahalawy/stroke-logistic-regression)

Ovde je takođe upotrebljena logistička regresija, sa drugačijom obradom podataka, i rezultati su sledeći:

```bash
Accuracy: 74.8472%

              precision    recall  f1-score   support

       False       0.98      0.75      0.85       929
        True       0.15      0.79      0.25        53

    accuracy                           0.75       982
   macro avg       0.57      0.77      0.55       982
weighted avg       0.94      0.75      0.82       982
```

Iako nije navedeno, pretpostavka je da je ovde prag podešen na 0.5, pa ću predstaviti i moje rezutate sa istim pragom, poređenja radi:

```bash
Accuracy: 77.1037%

              precision    recall  f1-score   support

       False       0.98      0.77      0.87       486
        True       0.14      0.72      0.24        25

    accuracy                           0.77       511
   macro avg       0.56      0.75      0.55       511
weighted avg       0.94      0.77      0.83       511
```

Rezultati su veoma slični, ali prag odlučivanja u ovom slučaju nije realan po mom mišljenju.
