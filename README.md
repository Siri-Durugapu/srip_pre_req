# Topic Classification Project

## Overview
This project is about building a text classification system that predicts the topic of a given text.  
All models used here are built from scratch (no pretrained models), following the given constraints.

---

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Siri-Durugapu/srip_pre_req.git
cd srip_pre_req
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

---

## Training

To train the model, run:

```bash
python src/train.py
```

This will preprocess the data, train the models, and save the final model.

---

## Inference

To test the model on custom input:

```bash
python src/inference.py 
```

Then enter your input when prompted.

Example:

```
Enter text: AI is changing the world. 
Predicted Topic: software_developmet
```

---

## Approach (Short Summary)

- Cleaned and preprocessed the text (lowercase, removed noise, etc.)
- Used TF-IDF for feature extraction:
  - word-level (unigrams + bigrams)
  - character-level (3–5 grams)
- Tried multiple models:
  - Naive Bayes
  - Logistic Regression + Active Learning
  - SVM
- Final model is a hybrid system combining all three using voting + a meta-model

---

## Results

- Best single model: SVM (~0.84 F1)
- Final hybrid model: ~0.82 F1

---

## Project Structure

```
project/
│── src/
│   ├── train.py
│   ├── inference.py
│   ├── model.py
│   └── utils.py
│── experiments/
│── final_models/
│── report.pdf
│── requirements.txt
│── README.md
```

---

## Reproducibility

- Random seed is fixed (42)
- No hardcoded paths are used
- Data sampling is deterministic
- The entire pipeline can be run using train.py and inference.py without manual changes

## Author

Siri Durugapu