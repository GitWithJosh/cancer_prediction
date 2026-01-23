# 🧠 Brain Tumor Classification System

Deep Learning System zur Klassifikation von Gehirntumoren aus MRI-Bildern mit Web-Interface.

## 📋 Übersicht

Automatische Klassifikation von MRI-Scans in 4 Kategorien mit CNN-Architekturen (VGG19, ResNet50, EfficientNet-B0) und Ensemble-Modell:

- 🔴 **Glioma**: Maligner Hirntumor
- 🟠 **Meningioma**: Benigner Tumor
- 🟡 **Pituitary**: Hypophysen-Tumor
- 🟢 **No Tumor**: Gesunder Scan

## 🚀 Quick Start

### 1. Conda-Umgebung erstellen

```bash
# Umgebung erstellen
conda create -n brain-tumor python=3.10 -y

# Aktivieren
conda activate brain-tumor

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Web-App starten

```bash
streamlit run app.py
```

Die App öffnet sich unter `http://localhost:8501`

### 3. Verwendung

1. Model auswählen (VGG19, ResNet50, EfficientNet-B0, Ensemble)
2. MRI-Bild hochladen (.jpg, .jpeg, .png)
3. "Classify Image" klicken
4. Ergebnisse ansehen (Klassifikation, Confidence, Wahrscheinlichkeiten)

## 📁 Projektstruktur

```
cancer_prediction/
├── cancer_prediction_images/     # Dataset (Training/Testing)
├── data/preprocessing.py         # Data Pipeline
├── models/architectures.py       # CNN Models
├── training/trainer.py           # Training Logic
├── evaluation/metrics.py         # Evaluation
├── app.py                        # Streamlit Web-App
├── config.yaml                   # Konfiguration
└── requirements.txt
```

## 🎯 Modelle trainieren (Optional)

```bash
# Alle Modelle trainieren
python training/train_all_models.py

# Oder einzeln in Python:
from data.preprocessing import load_and_split_dataset, create_data_loaders
from models.architectures import create_model
from training.trainer import train_model

train_data, val_data, test_data = load_and_split_dataset("cancer_prediction_images")
train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

model = create_model('resnet50', num_classes=4, pretrained=True)
trained_model, history = train_model(model, train_loader, val_loader, epochs=50)
```

## 📊 Model-Architekturen

- **VGG19**: 143M Parameter, Feature-Extraction eingefroren
- **ResNet50**: 25M Parameter, Residual Connections
- **EfficientNet-B0**: 5M Parameter, Compound Scaling
- **Ensemble**: Majority Voting über alle 3 Modelle

## ⚙️ Konfiguration

`config.yaml` anpassen:

```yaml
dataset:
  image_size: 224
  batch_size: 32
training:
  epochs: 5
  learning_rate: 0.001
```

## 🔬 Preprocessing

1. Resize auf 224×224
2. ImageNet-Normalisierung
3. Data Augmentation (Rotation, Flip, Color Jitter)

## ⚠️ Disclaimer

**WICHTIG**: Forschungs- und Bildungsprojekt. NICHT für klinische Diagnosen verwenden. Immer qualifizierte Ärzte konsultieren.

## 📝 Dokumentation

Siehe [API_DOCS.md](API_DOCS.md) für detaillierte API-Referenz.

## Overleaf Link

https://www.overleaf.com/8995259654wxvxkjpchwhj#1bab1b

## Datasets:

Der primäre Datensatz mit Test- und Trainingsdaten. <br>
Vier Klassen:  glioma, meningioma, no tumor, pituitary. <br>
5712 schwarz-weiß Bilder für Training, 1311 schwarz-weiß Bilder für Testing <br>
Die Bilder sind nicht alle gleich groß <br>
Paths: cancer_prediction_images/Training/* && cancer_prediction_images/Testing/*

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data