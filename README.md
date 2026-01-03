# 📰 Fake News Detection using Deep Learning (TensorFlow)

## 📌 Overview

In today’s hyperconnected digital ecosystem, the rapid spread of fake news poses serious risks ranging from public misinformation to political and social instability. This project focuses on building a **deep learning–based fake news detection system** that classifies news articles as **real or fake** based on their textual content.

The model leverages **natural language processing (NLP)** techniques combined with a **hybrid CNN–LSTM architecture** implemented in TensorFlow to capture both local textual patterns and long-term contextual dependencies.

---

## 🎯 Objectives

- Preprocess and clean raw news text data  
- Convert textual data into numerical representations using tokenization and padding  
- Utilize **pre-trained GloVe word embeddings** for semantic richness  
- Design and train a **CNN–LSTM neural network**  
- Evaluate model performance using accuracy, confusion matrix, and classification report  
- Demonstrate an end-to-end deep learning workflow for text classification  

---

## 📂 Dataset Description

- **Dataset name:** `fake_news.csv`  
- **Total samples:** 6,335  
- **Classes:**  
  - REAL  
  - FAKE  

### Features

| Column | Description |
|------|-------------|
| title | News headline |
| text  | Full news article |
| label | Target variable (REAL / FAKE) |

The dataset is well balanced across classes, making it suitable for supervised classification.

---

## 🛠 Tools & Technologies

- **Language:** Python  
- **Framework:** TensorFlow / Keras  
- **Libraries:**
  - NumPy  
  - Pandas  
  - scikit-learn  
  - Matplotlib  
  - TensorFlow (Keras API)  

- **NLP Techniques:**
  - Tokenization  
  - Sequence padding  
  - Pre-trained word embeddings (GloVe)  

---

## 🧠 Model Architecture

The model follows a **hybrid CNN–LSTM design**:

1. **Embedding Layer**
   - Initialized with pre-trained GloVe (300-dimensional)
   - Non-trainable to preserve semantic structure

2. **Dropout Layer**
   - Prevents overfitting

3. **1D Convolution Layer**
   - Extracts local n-gram patterns

4. **Max Pooling Layer**
   - Reduces dimensionality and computation

5. **LSTM Layer**
   - Captures long-term dependencies in text sequences

6. **Dense Output Layer**
   - Sigmoid activation for binary classification

Loss function: `binary_crossentropy`  
Optimizer: `Adam`

---

## 📁 Repository Structure

```
fake-news-detection/
│
├── data/
│ └── fake_news.csv
│
├── notebooks/
│ └── fake_news_detection.ipynb
│
├── results/
│ ├── fake_news_detection.html
│ └── fake_news_detection.pdf
│
├── LICENSE
└── README.md
```

---

## 📊 Results Summary

- The model achieves approximately 78% classification accuracy on the test set.
- Performance is balanced across REAL and FAKE classes.
- Training accuracy is high, while validation accuracy highlights realistic generalization behavior.
- Confusion matrix and learning curves provide transparent performance evaluation.
- Generated outputs are available in the results/ directory.

---

## 🧠 Key Insights

- Hybrid CNN–LSTM models effectively capture both local and sequential text features.
- Pre-trained embeddings significantly improve semantic understanding.
- Deep learning models can overfit quickly without regularization.
- Even with balanced data, validation performance reflects the inherent complexity of fake news detection.

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 📌 Notes

- This project is intended for **educational and portfolio purposes**.
- The code prioritizes **clarity, reproducibility, and interpretability**.
- Possible future extensions include:
  - Attention mechanisms
  - Bidirectional LSTM architectures
  - Handling class imbalance
  - Real-time news stream integration

🧾 Author
Mr Rup
GitHub: https://github.com/Mr-Rup

---
