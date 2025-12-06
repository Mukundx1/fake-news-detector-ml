# 📘 **TruthLens — Fake News Detection System**
*A Machine Learning + NLP powered web app to classify news as **Real** or **Fake**, with token-level explainability.*

TruthLens detects misinformation by analyzing linguistic patterns using TF-IDF vectorization and a linear ML model.  
It includes a clean Streamlit UI, built-in text cleaning, and transparent explainability.

---

## 🧠 **Project Motivation**
Fake news influences public opinion and spreads rapidly online.  
TruthLens aims to:

- Detect fake vs real news based purely on text  
- Provide explainability (why the model predicted Fake/Real)  
- Demonstrate practical NLP + ML techniques in an interactive app  

---

## 📂 **Dataset**

**Fake and Real News Dataset (Kaggle)**  
🔗 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
](url)
Dataset fields include:

- `title`  
- `text`  
- `subject`  
- `date`  
- `label` (Fake = 0, Real = 1)

Total records: ~44,000 articles.

---

## 🔍 **Exploratory Data Analysis (EDA)**

Key steps performed:

### ✔ Handling missing or unusable text  
- Removed URLs  
- Removed noise-only rows  
- Fallback to cleaned title when full text was empty  

### ✔ NLP Text Cleaning  
- Lowercasing  
- Removing punctuation  
- Removing URLs  
- Removing stopwords  
- Removing short tokens  
- Final feature: `clean_text_final`

### ✔ Class balance  
Fake and Real classes were nearly balanced — good for model training.

### ✔ Length & N-gram analysis  
Identified clickbait patterns and stylistic differences between Real vs Fake articles.

---

## 🤖 **Model Training**

### **TF-IDF Vectorizer**
- `max_features=20000`  
- `ngram_range=(1,2)`  
- English stopwords removed  

### **Classifier**
- Logistic Regression / PassiveAggressive Classifier (linear models)  
- Supports coefficient-based explainability  

### **Train/Test Split**
- 80% training  
- 20% testing  
- Stratified to maintain class balance  

---

## 📊 **Evaluation Metrics**

Metrics computed:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

## Accuracy: 0.91

Example:
<img width="1508" height="865" alt="Screenshot 2025-12-06 063738" src="https://github.com/user-attachments/assets/297429c2-5aa8-45b7-9ef2-c4e32d840ca6" />

