# mri-tumor-classification

Four-class brain tumor classification from MRI scans via fine-tuned CNNs and ensemble voting, with a Streamlit interface for single-image inference.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git_LFS-F64935?style=flat-square&logo=git&logoColor=white)

## Overview

Three pre-trained CNNs — VGG19, ResNet50, and EfficientNet-B0 — are fine-tuned on a dataset of 5,712 MRI images across four classes: glioma, meningioma, pituitary tumor, and no tumor. Each model is trained independently with ImageNet weights frozen during feature extraction and then released for fine-tuning. A majority-voting ensemble combines their predictions for the final classification. Model weights are stored via Git LFS.

## Models

| Model | Parameters | Architecture note |
|---|---|---|
| VGG19 | 143M | Deep feature extractor, frozen backbone |
| ResNet50 | 25M | Residual connections, strong baseline |
| EfficientNet-B0 | 5M | Compound scaling, most parameter-efficient |
| Ensemble | — | Majority vote across all three |

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 4 classes, 5,712 training images, 1,311 test images, all grayscale.

Place the dataset at `cancer_prediction_images/Training/` and `cancer_prediction_images/Testing/`.

## Quick Start

```bash
conda create -n mri-tumor python=3.10 -y
conda activate mri-tumor
pip install -r requirements.txt
streamlit run app.py
```

Upload a `.jpg` or `.png` MRI scan, select a model, and get a classification with confidence scores.

### Train Models (optional)

```bash
python training/train_all_models.py
```

Or individually:

```python
from data.preprocessing import load_and_split_dataset, create_data_loaders
from models.architectures import create_model
from training.trainer import train_model

train_data, val_data, test_data = load_and_split_dataset("cancer_prediction_images")
train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

model = create_model('resnet50', num_classes=4, pretrained=True)
trained_model, history = train_model(model, train_loader, val_loader, epochs=50)
```

Configuration is managed via `config.yaml` — adjust `image_size`, `batch_size`, `epochs`, and `learning_rate` there.

## Project Structure

```
mri-tumor-classification/
├── data/
│   └── preprocessing.py          # Dataset loading, augmentation, normalization
├── models/
│   └── architectures.py          # VGG19, ResNet50, EfficientNet-B0 factory
├── training/
│   ├── trainer.py
│   └── train_all_models.py
├── evaluation/
│   └── metrics.py
├── app.py                         # Streamlit inference UI
├── config.yaml
├── requirements.txt
└── cancer_prediction_images/      # Dataset (not tracked, use Git LFS for weights)
```

> **Note:** This project is for research and educational purposes. Not intended for clinical use.
