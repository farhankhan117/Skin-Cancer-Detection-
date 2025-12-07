# Skin Cancer Detection using Dual EfficientNetV2 + Metadata Fusion
Multi-Label Classification on ISIC 2019 / MILK10k Dataset

This repository contains an end-to-end deep learning pipeline for skin lesion classification using:

Dual-image inputs (Clinical + Dermoscopic)

Metadata fusion (age, sex, site, MONET features, etc.)

EfficientNetV2-S backbone

Multi-label classification (11 classes)

ISIC-formatted submission output

This project replicates a real ISIC challenge workflow and includes data preprocessing, model training, prediction, and submission generation.

# 📁 Dataset Overview

The project uses data from ISIC 2019 and MILK10k:

1. Training Ground Truth

Shape: 5240 × 12

Contains 11 diagnosis labels:

AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC

2. Metadata

Shape: 10480 × 17

Includes:

age_approx

sex

skin_tone_class

site

MONET concept features

image_type (clinical / dermoscopic)

# 3. Training Supplement

Additional diagnostic details

Shape: 10480 × 4

# 🔧 Preprocessing Workflow
1. Merge image paths with metadata

Clinical and dermoscopic images are linked using isic_id.

Missing image_type handled gracefully.

2. Pivot table creation

Each lesion is reshaped into:

lesion_id | clinical_image | dermoscopic_image

3. Filtering

Only lesions with both image types are kept → 5164 lesions.

4. Metadata Encoding

Age normalized (0–1)

Sex encoded (Female/Male)

Skin tone → one-hot (6 classes)

Site → one-hot (8 classes)

MONET features kept as continuous inputs

Total metadata features = 24
Final feature count including MONET = 31

# 🧠 Model Architecture
✔ Dual EfficientNetV2-S Backbones

One for clinical images

One for dermoscopic images

✔ Metadata Encoder

Fully connected network

✔ Fusion Layer

# Concatenates:

clinical_feats + dermoscopic_feats + metadata

✔ Classification Head

Linear layer

Outputs 11 sigmoid-activated diagnosis probabilities

✔ Loss Function

BCEWithLogitsLoss with aggressive class balancing for rare lesions.

✔ Optimizer

Adam(lr=1e-3)

# 📊 Training Details

80/20 train-validation split

Gradual unfreezing for fine-tuning

Gradient clipping to stabilize training

Epochs: 8

Batch size: 8

Best Validation Metrics

Macro F1: 0.2572

Accuracy: 75.94%

Best F1 classes:

BCC, NV, MEL, SCCKA

(Note: ISIC 2019 is highly imbalanced → low F1 is expected without larger training or stronger backbones.)

# 🧪 Test Inference

Test set: 479 lesions

Both image modalities detected automatically

Outputs raw logits → converted to sigmoid probabilities

Ensures all predictions are within [0,1]

📤 ISIC Submission Files

Two files are generated:

1. ISIC_submission_lesion_id.csv

Columns:

lesion_id, AKIEC, BCC, ..., VASC

2. ISIC_submission_image.csv

Columns:

image, AKIEC, BCC, ..., VASC


Both files:

Follow ISIC challenge format

Contain exactly 479 rows

No missing values

Probability-scaled outputs

# 🚀 Project Summary
Component	Value
Total lesions	5164
Validation lesions	1033
Test lesions	479
Metadata features	24
MONET features	7
Diagnosis outputs	11
Model parameters	40M+
Best Macro F1	0.2572
Best Accuracy	75.94%
# 📌 Folder Structure
/
├── data/
│   ├── train/
│   ├── test/
│   ├── metadata.csv
│   ├── ground_truth.csv
│
├── models/
│   └── skin_cancer_model.pth
│
├── notebooks/
│   └── training_pipeline.ipynb
│
├── submissions/
│   ├── ISIC_submission_lesion_id.csv
│   └── ISIC_submission_image.csv
│
└── README.md
