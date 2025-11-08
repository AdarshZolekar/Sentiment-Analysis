# Sentiment Analysis

A Python-based project that performs Sentiment Analysis on restaurant reviews using Natural Language Processing (NLP) techniques. The model classifies customer reviews as positive or negative using a supervised machine learning pipeline.

---

## Features

- Analyzes text sentiment from .tsv or .csv datasets

- Text preprocessing with tokenization, stopword removal and stemming

- Machine learning classification using Naive Bayes

- Generates accuracy score and confusion matrix

- Supports custom input review testing
 
- Built using Jupyter Notebook for interactive experimentation.

---

## How to Run 

1. **Clone the repository:**

```
   git clone https://github.com/AdarshZolekar/Sentiment-Analysis.git
```

2. **Run the script:** 

- Open a terminal or command prompt. 

- Navigate to the project directory:

```
   cd Sentiment-Analysis
```

- Run the Jupyter Notebook:

```
  jupyter notebook Sentiment-Analysis.ipynb
```

- Load the dataset (Restaurant-Reviews.tsv) and execute the notebook cells step-by-step.

---

## How It Works

1. Data Loading:
Reads the dataset containing restaurant reviews and their sentiment labels.

2. Text Preprocessing:

- Tokenization

- Removal of stopwords and punctuation

- Stemming using PorterStemmer

3. Feature Extraction:
Converts processed text into numerical form using Bag of Words model via CountVectorizer.

4. Model Training:
Trains a Naive Bayes Classifier on the processed dataset.

5. Evaluation:
Displays performance metrics such as accuracy and confusion matrix.

6. Prediction:
Allows users to input custom text reviews and view predicted sentiment.

---

## Dependencies

- pandas – Data handling

- numpy – Numerical computations

- nltk – Text preprocessing (tokenization, stemming, stopwords)

- scikit-learn – Model building and evaluation

- Jupyter Notebook – Interactive environment

Install them manually if needed:

```pip install pandas numpy nltk scikit-learn jupyter```

---

## Dataset

- File: ```Restaurant-Reviews.tsv```

- Description: Contains 1000 restaurant reviews labeled as 1 (positive) or 0 (negative).

- Columns:

  - ```Review``` – Customer feedback text.

  - ```Liked``` – Sentiment label (1 or 0).

---

## Results

- Algorithm Used: Naive Bayes

- Model Accuracy: ~73–78% (depending on dataset and preprocessing)

- Confusion Matrix: Displayed in notebook output.

---

## Future Improvements

- Implement deep learning models (LSTM, BERT) for better accuracy

- Add more sentiment categories (neutral, mixed)

- Build a web interface using Flask or Streamlit

- Use advanced NLP techniques like TF-IDF and word embeddings.

---

## License

This project is open-source under the MIT License.

---

## Contributions

Contributions are welcome!

- Open an issue for bugs or feature requests

- Submit a pull request for improvements.

<p align="center">
  <a href="#top">
    <img src="https://img.shields.io/badge/%E2%AC%86-Back%20to%20Top-blue?style=for-the-badge" alt="Back to Top"/>
  </a>
</p>


