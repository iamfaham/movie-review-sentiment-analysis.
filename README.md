# 🎬 Movie Review Sentiment Analysis

This project performs sentiment analysis on the IMDB movie reviews dataset, comparing traditional and deep learning models. It walks through an end-to-end NLP pipeline using both **Logistic Regression** (with TF-IDF) and **LSTM** (with embeddings + sequences).

## 📂 Dataset
- Source: [Kaggle IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Labels: `positive` or `negative`

## 🔧 Steps Covered
- Data collection with `kagglehub`
- Text cleaning (HTML tags, punctuation, stopwords)
- Feature extraction:
  - TF-IDF for Logistic Regression
  - Tokenization & Padding for LSTM
- Model training and evaluation:
  - `LogisticRegression` via scikit-learn
  - `LSTM` model using TensorFlow/Keras
- Visualizations (accuracy/loss curves)
- Metrics: Accuracy, Confusion Matrix, Classification Report

## 🧠 Results
- Logistic Regression: ~**89.2%**
- LSTM Model: ~**88.0%**

## 📌 Conclusion
While LSTM offers the potential for deeper language understanding, the Logistic Regression model performs surprisingly well given its simplicity — highlighting the power of good preprocessing and feature engineering.

## 🚀 Run the Notebook
Make sure the following libraries are installed:
```bash
pip install pandas numpy nltk scikit-learn tensorflow kagglehub matplotlib seaborn
```
📁 File
- movie_review_sentiment_analysis.ipynb: Jupyter Notebook with full pipeline and analysis.

Feel free to star 🌟 this repo if you found it helpful!
