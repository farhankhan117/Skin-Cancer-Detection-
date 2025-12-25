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
