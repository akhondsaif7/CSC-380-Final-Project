# Explainable Detection of Online Sexism (EDOS) — Binary Classification

## Highlights
- Built a **hybrid ensemble** combining a Random Forest classifier with a
  fine-tuned RoBERTa transformer for binary sexism detection in tweets
- Achieved a **weighted F1-score of ~0.85**, exceeding the 0.82 threshold
- Based on [SemEval 2023 - Task 10](https://codalab.lisn.upsaclay.fr/competitions/7124#learn_the_details-overview)

## Overview
Online sexism is a widespread and harmful phenomenon. This project tackles
**Task A — Binary Sexism Detection**, classifying social media posts as
either *sexist* or *not sexist*. It was completed as the final project
for CSC380.

We ensembled two complementary approaches:
1. **Random Forest** on character-level TF-IDF (3–5 char n-grams)
2. **Fine-tuned RoBERTa** (`cardiffnlp/twitter-roberta-base-offensive`)

The final prediction averages the predicted probabilities from both
models with a tuned decision threshold.

## Dataset
- Subset of the EDOS labelled dataset (~5k samples)
- Columns used: `text`, `label`, `split`
- Labels: `sexist`, `not sexist`

> **Note:** The dataset file (`edos_labelled_data.csv`) should be placed
> in the project root directory.

## Preprocessing
- URLs replaced with `URL` placeholder
- @mentions replaced with `USER` placeholder
- All text lowercased

## Models & Results

| Feature + Model                          | Sexist F1 | Non-Sexist F1 | Weighted F1 |
|------------------------------------------|-----------|---------------|-------------|
| BoW + Logistic Regression                | 0.54      | 0.86          | 0.77        |
| Char TF-IDF + Linear SVM                | 0.54      | 0.86          | 0.77        |
| Char TF-IDF + Random Forest             | 0.59      | 0.88          | 0.80        |
| **Char TF-IDF + RF & RoBERTa (Ensemble)** | **0.72** | **0.90**      | **0.85**    |

## Installation & Usage

### Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas, numpy

### Install dependencies
```bash
pip install torch transformers scikit-learn pandas numpy
