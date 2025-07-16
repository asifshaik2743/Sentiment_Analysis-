---

# 🧠 Sentiment Analysis on Product Reviews

This project performs **Sentiment Analysis** on product reviews using machine learning and NLP (Natural Language Processing) techniques. The goal is to classify the sentiment of a review as either **positive** or **negative**.

---

## 📁 Project Structure

```
.
├── Reviews.csv                     # Dataset containing product reviews
├── Sentiment_Analysis_code.ipynb  # Jupyter notebook with data analysis and model training
├── requirements.txt               # List of Python dependencies
├── README.md                      # Project documentation
```

---

## 📊 Dataset

* **File:** `Reviews.csv`
* **Description:** Contains text reviews with associated metadata. Only relevant fields (such as review text and sentiment label) are used in this project.

---

## 🔄 Workflow

The main steps in the project are:

1. **Data Preprocessing**

   * Handle missing data
   * Normalize text (lowercase, remove stopwords, lemmatization)
   * Tokenize using `nltk` and `spaCy`

2. **Exploratory Data Analysis (EDA)**

   * Visualize most frequent words
   * Plot sentiment distribution

3. **Modeling**

   * Convert text to numerical features using `TF-IDF`
   * Train model using `Logistic Regression`
   * Evaluate using accuracy, precision, recall, and F1-score

4. **Visualization**

   * Confusion matrix
   * Word clouds and other charts

---

## 🚀 How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Open and run the notebook**

```bash
jupyter notebook Sentiment_Analysis_code.ipynb
```

---

## 🧾 Requirements

Listed in `requirements.txt`:

* pandas==2.0.3
* numpy==1.24.4
* scikit-learn==1.3.0
* nltk==3.8.1
* spacy==3.6.0
* matplotlib==3.7.2
* seaborn==0.12.2
* jupyter==1.0.0

To install them all:

```bash
pip install -r requirements.txt
```

Also, make sure to download necessary resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

And install spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

## 📌 Notes

* This project is designed for educational purposes.
* The dataset used is assumed to be clean and in `.csv` format with text-based sentiment labels.

---

## 📄 License

This project is open-source and free to use for non-commercial purposes.

---
