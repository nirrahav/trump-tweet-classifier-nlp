# Trump Tweet Authorship Classification 🐦🇺🇸

This project explores the task of **authorship classification** on tweets associated with former U.S. President Donald Trump. The goal is to determine whether a tweet was personally written by Trump or by someone from his team, based on textual and metadata features.

## 🔍 Project Overview

Using a labeled dataset of Trump-related tweets, we developed a full NLP pipeline including:
- Advanced **text preprocessing**
- Custom **feature engineering**
- Multiple **machine learning models**
- Comparison of traditional ML models vs deep learning (FFNN, BERT)

## 📊 Models Trained
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Feedforward Neural Network (FFNN)
- BERT (via Hugging Face Transformers)

## 🧠 Feature Engineering
- **TF-IDF** & **Word2Vec** text embeddings
- Metadata: time, device, punctuation count, word stats
- Boolean features: use of capital letters, holidays, etc.

## 🛠️ Tools & Libraries
- Python, NumPy, pandas, scikit-learn
- Matplotlib, Seaborn
- XGBoost, PyTorch
- Hugging Face Transformers
- Optuna (for hyperparameter tuning)

## ⚙️ Preprocessing
- Contraction expansion (`don't` → `do not`)
- Special character removal
- Lemmatization & stopword filtering
- Metadata normalization (date, time, device)

## 🧪 Evaluation Metrics
Each model was evaluated on:
- **Accuracy**
- **F1-Score**
- **Precision**
- **Recall**

## 🏆 Key Results
| Model      | Accuracy | F1  | Recall | Precision |
|------------|----------|-----|--------|-----------|
| BERT       | 0.88     | 0.81| 0.71   | 0.94      |
| XGBoost    | 0.89     | 0.83| 0.78   | 0.88      |
| SVM        | 0.82     | 0.73| 0.69   | 0.79      |
| FFNN       | 0.80     | 0.72| 0.69   | 0.75      |
| LogReg     | 0.86     | 0.79| 0.73   | 0.87      |

## 📁 Files
- `trump_tweet_authorship_classification.ipynb`: Main notebook
- `Answers.pdf`: Final report (in Hebrew)
- `README.md`: This file 😊

## 👤 Authors
- Nir Rahav  
- Yael Berkovich

## 📌 Notes
This project was developed as part of the course *Introduction to Natural Language Processing* at Ben-Gurion University.

# Trump Tweet Authorship Classification 🐦🇺🇸

This project explores the task of **authorship classification** on tweets associated with former U.S. President Donald Trump. The goal is to determine whether a tweet was personally written by Trump or by someone from his team, based on textual and metadata features.

## 🔍 Project Overview

Using a labeled dataset of Trump-related tweets, we developed a full NLP pipeline including:
- Advanced **text preprocessing**
- Custom **feature engineering**
- Multiple **machine learning models**
- Comparison of traditional ML models vs deep learning (FFNN, BERT)

## 📊 Models Trained
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Feedforward Neural Network (FFNN)
- BERT (via Hugging Face Transformers)

## 🧠 Feature Engineering
- **TF-IDF** & **Word2Vec** text embeddings
- Metadata: time, device, punctuation count, word stats
- Boolean features: use of capital letters, holidays, etc.

## 🛠️ Tools & Libraries
- Python, NumPy, pandas, scikit-learn
- Matplotlib, Seaborn
- XGBoost, PyTorch
- Hugging Face Transformers
- Optuna (for hyperparameter tuning)

## ⚙️ Preprocessing
- Contraction expansion (`don't` → `do not`)
- Special character removal
- Lemmatization & stopword filtering
- Metadata normalization (date, time, device)

## 🧪 Evaluation Metrics
Each model was evaluated on:
- **Accuracy**
- **F1-Score**
- **Precision**
- **Recall**

## 🏆 Key Results
| Model      | Accuracy | F1  | Recall | Precision |
|------------|----------|-----|--------|-----------|
| BERT       | 0.88     | 0.81| 0.71   | 0.94      |
| XGBoost    | 0.89     | 0.83| 0.78   | 0.88      |
| SVM        | 0.82     | 0.73| 0.69   | 0.79      |
| FFNN       | 0.80     | 0.72| 0.69   | 0.75      |
| LogReg     | 0.86     | 0.79| 0.73   | 0.87      |

## 📁 Files
- `trump_tweet_authorship_classification.ipynb`: Main notebook
- `Answers.pdf`: Final report (in Hebrew)
- `README.md`: This file 😊

## 👤 Authors
- Nir Rahav  
- Yael Berkovich

## 📌 Notes
This project was developed as part of the course *Introduction to Natural Language Processing* at Ben-Gurion University.

