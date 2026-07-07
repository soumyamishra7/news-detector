# 📰 Fake News Detection System

An end-to-end Machine Learning web application that classifies news headlines/articles as **Real** or **Fake** using Natural Language Processing and classical supervised learning models — trained, evaluated, and deployed as a live Flask web app.

🔗 **Live Demo:** _(https://news-detector-mykz.onrender.com)_
📁 **Repository:** https://github.com/soumyamishra7/fake-news-detection-system

---

## 🚀 Features

- Paste any news headline or article and get an instant **Real / Fake** prediction
- Confidence score shown alongside the prediction
- Displays the **top keywords** influencing the model's decision (explainability)
- Compares three ML models — **Logistic Regression, Naive Bayes, and SVM**
- Clean, responsive web UI (HTML, CSS, JavaScript)

---

## 🧠 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 98.67%   | 98.26%    | 98.97% | 98.61%   |
| Naive Bayes         | 92.88%   | 92.83%    | 92.15% | 92.49%   |
| **SVM (best)**       | **99.33%** | **99.27%** | **99.32%** | **99.30%** |

The **SVM** model was selected for deployment based on its superior accuracy and balanced precision-recall trade-off.

---

## 🏗️ System Architecture

**Two layers — trained offline, served online:**

**1. Offline (Training / Notebook)**
`Fake.csv + True.csv (44,898 articles)` → Clean & merge → `TF-IDF vectorizer (5,000 features)` → Train & compare (LR vs NB vs SVM) → Save best model (`joblib.dump`)

**2. Online (Flask app on Render)**
User pastes headline → Flask vectorizes input → Model predicts label → Result + top keywords shown to user

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **ML/NLP:** Scikit-learn, Pandas, NumPy, Joblib
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle, by Clément Bisaillon)
- **Deployment:** Render
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
fake-news-detection-system/
├── app.py                  # Flask application (routes + inference)
├── detector.py             # Prediction/inference logic
├── train_model.py          # Model training script
├── news_analysis.ipynb     # EDA & model comparison notebook
├── Fake.csv                # Raw fake news articles
├── True.csv                # Raw real news articles
├── fake_or_real_news.csv   # Cleaned/merged dataset
├── model.pkl               # Serialized trained model (SVM)
├── vectorizer.pkl          # Serialized TF-IDF vectorizer
├── word_scores.pkl         # Word importance scores (explainability)
├── templates/              # HTML templates for the Flask app
├── presentation/           # Project presentation slides
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/soumyamishra7/fake-news-detection-system.git
cd fake-news-detection-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app locally
```bash
python app.py
```
The app will be available at `http://127.0.0.1:5000`

### 4. (Optional) Retrain the model
```bash
python train_model.py
```

---

## 📊 How It Works

1. **Data Preprocessing** — Clean and merge the Fake/Real datasets, remove punctuation, stop-words, and normalize text.
2. **Feature Engineering** — Convert cleaned text into numerical vectors using TF-IDF (top 5,000 features).
3. **Model Training** — Train and compare Logistic Regression, Naive Bayes, and SVM classifiers.
4. **Evaluation** — Assess models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
5. **Deployment** — Serialize the best model with Joblib and serve predictions through a Flask app hosted on Render.

---

## 🔮 Future Improvements

- Incorporate transformer-based embeddings (e.g., BERT) for deeper semantic understanding
- Support multi-lingual fake news detection
- Real-time integration with live news feeds / social media APIs
- Source-credibility scoring
- Browser extension for on-the-fly article checking

---

## 👤 Author

**Soumya Ranjan Mishra**
MCA, Rourkela Institute of Management Studies (RIMS)
Project Guide: Prof. Bibhudendu Panda, Sr. Faculty, RIMS

---

## 📄 License

This project is for academic purposes as part of an MCA internship/project submission.
