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

Iz **root** direktorijuma:

- Pripremite podatke:
  
  ```bash
  python3 src/prepare_data.py
  ```

- Trenira i validira model:
  
  ```bash
  python3 src/train_and_validate_model.py
  ```

- Testira model:
  
  ```bash
  python3 src/test_model.py
  ```

Ovo će:
- Predobraditi i podeliti podatke
- Trenirati prilagođenu logističku regresiju
- Sačuvati težine modela, bias i scaler
- Testirati vaš model

## Pravljenje predikcije

Da biste predvideli verovatnoću moždanog udara za novog pacijenta iz terminala:

```bash
python3 src/stroke_predictor.py
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

- Prag je podešen na 0.9 jer je mnogo verovatnije da negativni primeri imaju visoke predviđene verovatnoće nego da pozitivni primeri imaju niske  
- U ovom skupu podataka postojao je jedan primer gde je pol bio „Other“, koji je uklonjen da bi se poboljšale performanse treninga  
