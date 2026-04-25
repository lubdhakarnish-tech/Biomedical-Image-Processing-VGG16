# Biomedical Image Processing using VGG16

## 📌 Project Overview
This project analyzes the performance of VGG16 CNN for image classification on both medical and non-medical datasets.

## 🎯 Objectives
- Implement VGG16 for image classification
- Compare:
  - Training from scratch
  - Pretrained model
  - Fine-tuned model
- Evaluate accuracy and training time

## 📂 Datasets Used
1. Chest X-ray Dataset (Medical)
   - Classes: NORMAL, PNEUMONIA

2. Cats vs Dogs Dataset (Non-medical)
   - Classes: Cat, Dog

## 🧠 Models Implemented
- VGG16 (from scratch)
- Pretrained VGG16 (frozen layers)
- Fine-tuned VGG16

## 📊 Results Summary
| Model | Accuracy |
|------|--------|
| From Scratch | ~62% |
| Pretrained | ~86% |
| Fine-tuned | ~88% |

## 🔍 Key Observations
- Pretrained models perform best on small datasets
- Fine-tuning improves accuracy but may cause overfitting
- Medical images are harder than natural images

## ⚠️ Limitations
- Overfitting in fine-tuned model
- Small dataset size
- Limited generalization

## 🚀 Future Work
- Apply transfer learning to psoriasis severity (PASI scoring)
- Improve generalization using regularization
- Explore explainable AI techniques

## 📎 Files
- `main.py` → Clean implementation
- `notebook.ipynb` → Experimental workflow and results
- `presentation.pdf` → Project presentation

## 🛠️ Requirements
See `requirements.txt`

## 📌 Author
Lubdhak Nairith Saha Arnish
