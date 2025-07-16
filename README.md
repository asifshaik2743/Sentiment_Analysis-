---

# 🧠 Sentiment Analysis on Product Reviews

This project is an end-to-end **Sentiment Analysis** application that uses machine learning and natural language processing (NLP) techniques to predict the sentiment (positive/negative) from product review text.

## 📁 Project Structure

```
.
├── Reviews.csv                     # Dataset containing product reviews
├── Sentiment_Analysis_code.ipynb  # Jupyter notebook with full analysis and model training
├── requirements.txt               # Required Python dependencies
├── README.md                      # Project documentation (you are here)
```

---

## 📊 Dataset

* **Source:** `Reviews.csv`
* **Description:** Contains product reviews data including the review text and associated metadata.
* **Usage:** Only the text data and corresponding sentiment labels were used for training the model.

---

## ⚙️ Features & Workflow

The project includes:

1. **Data Loading & Preprocessing**

   * Handling missing values
   * Text normalization using `nltk` and `spaCy`
   * Tokenization, stop word removal, and lemmatization

2. **Exploratory Data Analysis (EDA)**

   * Word frequency analysis
   * Sentiment distribution visualization using `matplotlib` and `seaborn`

3. **Model Training**

   * Vectorization using `TfidfVectorizer`
   * Model training using `Logistic Regression`
   * Evaluation using accuracy, precision, recall, and F1-score

4. **Visualization**

   * Confusion matrix and performance plots

---

## 🛠️ Installation

1. Clone the repository or download the files.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Project

Open the Jupyter notebook and run all cells sequentially:

```bash
jupyter notebook Sentiment_Analysis_code.ipynb
```

---

## 📦 Requirements

* Python 3.8+
* All Python packages listed in `requirements.txt`:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `nltk`
  * `spacy`
  * `matplotlib`
  * `seaborn`
  * `jupyter`

You can install them all via:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* Be sure to download `nltk` data packages before running:

  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```
* For `spaCy`, ensure the English language model is installed:

  ```bash
  python -m spacy download en_core_web_sm
  ```

---

## 📚 License

This project is for educational and non-commercial purposes.

---
