# 🩺 Skin Cancer Detection with Dual-Modality Deep Learning
 ## 📋 Project Overview

This project presents a deep learning–based skin cancer detection system for multi class skin lesion diagnosis using the MILK10k / ISIC 2025 dataset.
The model jointly analyzes dual modality images clinical close-up and dermoscopic along with patient metadata to classify lesions into 11 diagnostic categories.

By fusing visual features with clinical context, the system improves diagnostic robustness and outputs class wise probability scores, enabling reliable evaluation using the Macro F1 Score on a blind benchmark test set.

Performance Achieved (ISIC MILK10k Challenge):

Accuracy: 75.94%
Macro F1 Score: 0.2572

## 📊 Dataset Summary
### Training Dataset (MILK10k)

Lesions: 5,240  
Images: 10,480 JPEG images  
1 clinical close-up image per lesion  
1 dermatoscopic image per lesion  
Metadata entries: 10,480  
Supplemental metadata entries: 10,480  
Ground truth labels: 5,240 lesion diagnoses  

### Benchmark (Test) Dataset

Lesions: 479  
Images: 958 JPEG images (clinical + dermoscopic pairs)  
Metadata entries: 958  
Ground truth: Hidden (used for leaderboard evaluation)  


## 🧬 Diagnostic Categories (11 Classes)

Each lesion is classified into one of 11 diagnostic categories, and the model outputs probability scores for all classes.

| Abbreviation | Diagnostic Category |
|-------------|---------------------|
| AKIEC | Actinic keratosis / intraepidermal carcinoma |
| BCC | Basal cell carcinoma |
| BEN_OTH | Other benign proliferations (including collision tumors) |
| BKL | Benign keratinocytic lesion |
| DF | Dermatofibroma |
| INF | Inflammatory and infectious conditions |
| MAL_OTH | Other malignant proliferations |
| MEL | Melanoma |
| NV | Melanocytic nevus |
| SCCKA | Squamous cell carcinoma / keratoacanthoma |
| VASC | Vascular lesions and hemorrhage |


## 📥 Input Data Description

Each lesion includes the following components:

### 1️⃣ Dual-Modality Image Pair
- Clinical close-up image  
- Dermatoscopic image  

### 2️⃣ MONET Concept Annotations
Probability scores are provided for:
- Ulceration / crust  
- Hair  
- Vasculature  
- Erythema  
- Pigmentation  
- Gel / dermoscopy liquid  
- Skin markings (pen ink, purple pen)  

### 3️⃣ Additional Metadata
- Age (grouped in 5-year intervals)  
- Sex  
- Skin tone (0 = very dark → 5 = very light)  
- Anatomical site


## ⚙️ Data Processing Pipeline

**Input:** Raw images + CSV metadata  
**Output:** Clean dataset with **5,164 lesion image pairs**

### Key Processing Steps
- Extract image archives  
- Pair clinical and dermoscopic images  
- Merge metadata (age, sex, skin tone, anatomical site)  
- Add MONET concept annotations  
- Handle missing values:
  - Median filling  
  - One-hot encoding  
- Stratified train–validation split (80–20)

### Dataset Split
- **Training:** 4,131 samples (80%)  
- **Validation:** 1,033 samples (20%)  

## 🧩 Custom Dataset & Augmentation

### Custom Dataset Class
- **SkinLesionDataset**

### Image Transforms

**Training:**
- Resize (256 × 256)  
- Random flip  
- Rotation  
- Color jitter  
- Normalization  

**Validation:**
- Resize  
- Normalization only  


## 🧠 Model Architecture  
### Dual-EfficientNetV2-S with Metadata Fusion

- **Backbone:** Two EfficientNetV2-S models (ImageNet pretrained)

### Input
- **Clinical image:** 256 × 256 RGB  
- **Dermatoscopic image:** 256 × 256 RGB  
- **Metadata:** 24 features  
  - Age  
  - Sex  
  - Skin tone  
  - Anatomical site  
  - 7 MONET concept features  

### Fusion Strategy
- Feature concatenation of:
  - Clinical image embeddings  
  - Dermoscopic image embeddings  
  - Metadata features  
- Followed by a linear classifier  

### Output
- **11-class probability vector** (multi-label format)

### Model Size
- **Total parameters:** ~40.3M  
- **Initially trainable parameters:** ~30.4K (classifier only)




## ⚖️ Handling Class Imbalance

### Loss Function
- **BCEWithLogitsLoss** with class-wise positive weights

### Training Strategy
-Train classifier only (backbone frozen)  
-Unfreeze backbone  
-Full fine-tuning with early stopping  

---

## 📏 Evaluation Metric

### Primary Metric: Macro F1 Score
- Computes F1 score independently for each class  
- Averages scores equally across all classes  
- Robust to severe class imbalance  

### Thresholding
- Predictions ≥ 0.5 are considered positive  
- A lesion may be predicted positive for multiple classes  

---

## 🎯 Final Results

- **Macro F1 Score:** 0.2572  
- **Overall Accuracy:** 75.94%  
- **Validation Loss:** 13.2619  

### Per-Class F1 Scores

| Class | F1 Score | Samples |
|------|----------|---------|
| BCC | 0.7819 | 498 |
| NV | 0.5911 | 148 |
| MEL | 0.2963 | 88 |
| SCCKA | 0.3444 | 92 |
| AKIEC | 0.2167 | 60 |
| BKL | 0.2063 | 107 |
| DF | 0.1227 | 10 |
| VASC | 0.1007 | 9 |
| INF | 0.0930 | 10 |
| BEN_OTH | 0.0755 | 9 |

---

## 🧪 Test Inference (Steps 14–15)

- **Test lesions:** 479 (blind benchmark)  
- Missing metadata handled with default values  
- Batch inference with **batch size = 16**  
- Predictions submitted for leaderboard evaluation  

---

## 📝 Dataset Notes

- Benchmark dataset includes **new ISIC-DX diagnoses** not present in the original MILK10k dataset  
- Some original diagnostic categories are omitted, particularly:
  - Other benign  
  - Other malignant  
- Granular ISIC-DX labels are available **only in training supplemental metadata**  

---

## 📜 License

- **License:** CC-BY-NC (Attribution, Non-Commercial)

