# Fake Review Detection using Supervised Machine Learning

This repository contains the implementation of a supervised machine learning framework to detect fake online hotel reviews. Our goal is to identify deceptive (fake) reviews using content-based features and classify them using traditional ML algorithms.

## 📄 Overview

Online reviews significantly influence consumer decisions. Unfortunately, some reviews are intentionally deceptive, posted to mislead buyers by either promoting or demoting a product or service. This project applies supervised learning methods to detect such fake reviews based on linguistic and psychological features.

We use a gold-standard dataset introduced by Ott et al., which includes both truthful and deceptive reviews across positive and negative sentiments.

## 📊 Features Used

We extract and use the following content-based features for classification:

- **TF-IDF**: Term frequency–inverse document frequency to weigh words.
- **Empath Categories**: Psycholinguistic features derived using the Empath tool, similar to LIWC.
- **Sentiment Polarity**: Sentiment score (positive/negative) provided in the dataset.

## 🧠 Machine Learning Models

We experimented with the following supervised classifiers:

- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

These models are trained and evaluated on a balanced dataset of fake and truthful hotel reviews.

## 📁 Dataset

The dataset used is the benchmark dataset by Ott et al. ([Ott et al., 2013](https://aclanthology.org/P11-1033.pdf)), which consists of 1600 hotel reviews:

- 800 deceptive (fake)
- 800 truthful
- Balanced in sentiment (positive/negative)

Each review is labeled as:
- `1` – Truthful
- `0` – Deceptive

> Note: The dataset is not included in this repository due to licensing restrictions. Please obtain it from [Ott et al.’s official website](http://myleott.com/op-spam.html).

## 📊 Results

Our best performance was achieved using the Support Vector Machine (SVM) classifier with TF-IDF + Empath + Sentiment features, achieving over **86% accuracy**.

Model performance metrics include:
- Accuracy
- Precision
- Recall
- F1-score

Refer to the `results/` folder for detailed evaluation reports and confusion matrices.

## 📦 Project Structure

```plaintext
.
├── data/
│   └── [Dataset files, organized into train/val/test]
├── feature_extraction/
│   ├── tfidf_extractor.py
│   ├── empath_features.py
│   └── sentiment_analysis.py
├── models/
│   ├── logistic_regression.py
│   ├── naive_bayes.py
│   └── svm_classifier.py
├── results/
│   └── evaluation_metrics.csv
├── utils/
│   └── preprocessing.py
├── main.py
└── README.md

