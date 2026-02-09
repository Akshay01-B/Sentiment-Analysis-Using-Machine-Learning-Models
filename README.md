## 📊 Sentiment Analysis Using Machine Learning
## 📌 Project Overview

This project builds a multi-class Sentiment Analysis system using Machine Learning techniques to classify social media text into different sentiment categories such as Positive, Joy, Excitement, Neutral, Sad, Gratitude, and more.

The project demonstrates a complete NLP and ML pipeline — from data preprocessing to model evaluation and prediction.

## 🎯 Objectives

Perform sentiment classification on social media text

Compare multiple ML models

Identify the best-performing model

Build a reusable sentiment prediction pipeline

## 📂 Dataset

The dataset contains 732 social media posts with features such as:

Text (main input)

Sentiment (target label)

Timestamp

User

Platform (Twitter, Instagram, Facebook)

Hashtags

Likes & Retweets

Country

Date & Time features

Key Observations

No missing values

Highly imbalanced sentiment classes

Large number of unique sentiment labels

## 🛠️ Tech Stack

Python

Pandas & NumPy

Matplotlib & Seaborn

NLTK

Scikit-learn

WordCloud

Joblib

## 🔍 Project Workflow
1️⃣ Data Loading & Inspection

Loaded CSV dataset

Checked data types, summary stats, and missing values

2️⃣ Exploratory Data Analysis (EDA)

Sentiment distribution analysis

Platform-wise and country-wise sentiment analysis

Likes vs Retweets comparison

Hourly sentiment trends

3️⃣ Text Preprocessing

Lowercasing text

Removing URLs, mentions, and hashtags

Removing punctuation and special characters

Stopword removal using NLTK

4️⃣ Feature Engineering

TF-IDF Vectorization

N-grams (1–3)

Rare class grouping into "Other"

5️⃣ Model Training

Models tested:

Logistic Regression

Naive Bayes

Linear SVM

Random Forest

6️⃣ Model Evaluation

Accuracy comparison

Classification report

Best model selection

## 📈 Results
Model	Accuracy
Logistic Regression	21.77%
Naive Bayes	19.73%
Random Forest	43.54%
⭐ Linear SVM	46.26%

## ✅ Best Model: Linear SVM

🧠 Sample Predictions
Text	Predicted Sentiment
I love this new feature!	Love
The service was awful and slow	Disappointed
It's okay, not too bad	Bad
## 💾 Saved Files

cleaned_sentiment_dataset.csv

## 📚 Key Learnings

Sentiment data is often imbalanced

Preprocessing strongly impacts performance

Too many sentiment classes reduce accuracy

SVM works well for text classification

## ⚠️ Limitations

Moderate accuracy due to many classes

Class imbalance affects results

Traditional ML struggles with nuanced emotions

## 🚀 Future Improvements

Use Deep Learning (LSTM, BERT)

Apply class balancing techniques

Hyperparameter tuning

Larger dataset

Deploy as API or web app

## Conclusion

This project successfully builds a multi-class sentiment analysis
system using machine learning on social media text data. 
It demonstrates key NLP steps such as text preprocessing, TF-IDF feature extraction, 
and model comparison. Linear SVM achieved the best performance, 
showing the effectiveness of traditional ML for text classification. 
The project also highlights real-world challenges like 
class imbalance and many sentiment categories, 
while providing a strong foundation for future improvements with advanced NLP models.
