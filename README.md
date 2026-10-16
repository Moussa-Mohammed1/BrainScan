# BrainScan

BrainScan est un projet de classification d'images d'IRM cérébrales par
réseau de neurones convolutif (CNN). Il utilise TensorFlow/Keras pour le
modèle, OpenCV pour le chargement et le prétraitement des images, et
Streamlit pour tester une prédiction depuis une interface web.

Le dataset doit être organisé en un dossier par classe, par exemple :

```text
data/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

Les noms des dossiers sont utilisés comme labels. Les images sont redimensionnées
en `224 x 224` pixels, converties en RGB et normalisées dans `[0, 1]`.

## Structure du projet

```text
BrainScan/
├── app/
│   └── streamlit_app.py       # Interface web de prédiction
├── data/                      # Dataset local, non versionné
│   └── processed/             # Tableaux NumPy prétraités, non versionnés
├── models/                    # Modèles et artefacts générés, non versionnés
├── notebooks/
│   ├── 01_exploration.ipynb  # Exploration et préparation du dataset
│   └── 02_cnn_architecture.ipynb
├── src/
│   ├── config.py              # Chemins et hyperparamètres
│   ├── data_prep.py           # Chargement, encodage et séparation des données
│   ├── evaluate.py            # Évaluation, rapports et visualisations
│   ├── model.py               # Architecture, compilation et sauvegarde du modèle
│   ├── predict.py             # Prédiction d'une image
│   ├── train.py               # Entraînement et checkpoint du meilleur modèle
│   └── utils.py               # Chargement des données prétraitées
├── requirements.txt
└── README.md
```

## Installation

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows, activez l'environnement avec :

```powershell
.venv\Scripts\Activate.ps1
```

## Préparer les données

Placez le dataset dans `data/`, avec un sous-dossier par classe. La préparation
peut être utilisée depuis Python :

```python
from src.data_prep import prepare_dataset

dataset = prepare_dataset()
X_train = dataset["X_train"]
X_test = dataset["X_test"]
y_train = dataset["y_train"]
y_test = dataset["y_test"]
label_encoder = dataset["label_encoder"]
```

Les notebooks peuvent également être utilisés pour l'exploration et pour
exporter les tableaux dans `data/processed/`.

## Entraîner et sauvegarder le modèle

Le code d'entraînement utilise `model.fit()` et sauvegarde automatiquement le
meilleur modèle selon `val_loss` dans :

```text
models/brainscan_best.keras
```

Pour produire l'artefact utilisé par l'application Streamlit, sauvegardez aussi
le modèle et l'encodeur de labels :

```python
from src.data_prep import prepare_dataset
from src.model import save_trained_model
from src.train import save_label_encoder, train_model

dataset = prepare_dataset()
model, history, duration = train_model(data=dataset)

save_trained_model(model)                 # models/brainscan_model.h5
save_label_encoder(dataset["label_encoder"])  # models/label_encoder.pkl
```

Le fichier `models/brainscan_model.h5` est le modèle chargé par l'interface de
déploiement. Ces fichiers sont générés localement et sont exclus du dépôt Git.

## Évaluer le modèle

Les fonctions d'évaluation sont disponibles dans `src/evaluate.py` :

```python
from src.evaluate import (
    evaluate_model,
    get_classification_report,
    get_confusion_matrix,
    plot_prediction_examples,
    plot_training_history,
)

metrics = evaluate_model(model, X_test, y_test)
report = get_classification_report(
    model,
    X_test,
    y_test,
    class_names=list(label_encoder.classes_),
)
matrix = get_confusion_matrix(model, X_test, y_test)

plot_training_history(history)
plot_prediction_examples(
    model,
    X_test,
    y_test,
    class_names=list(label_encoder.classes_),
)
print(metrics)
print(report)
print(matrix)
```

Ces fonctions permettent d'obtenir les métriques de test, les courbes
d'apprentissage, la matrice de confusion, le rapport de classification et des
exemples de prédictions correctes ou incorrectes.

## Tester une image

Une image peut être prédite directement avec `predict_image()` :

```python
from src.predict import predict_image

label, probabilities = predict_image(
    "chemin/vers/image.jpg",
    model,
    label_encoder,
)
print(label)
print(probabilities)
```

## Lancer l'interface Streamlit

Après avoir généré `models/brainscan_model.h5` et
`models/label_encoder.pkl`, lancez :

```bash
streamlit run app/streamlit_app.py
```

L'interface accepte les formats `.jpg`, `.jpeg`, `.png` et `.bmp`, affiche
l'image importée, la classe prédite et les probabilités de chaque classe.
