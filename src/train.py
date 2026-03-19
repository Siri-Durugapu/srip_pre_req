import time
import numpy as np
import pickle
from scipy.sparse import vstack, hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from utils import load_data, preprocess_text
from model import train_nb, train_lr, train_svm

df = load_data()
df["DATA"] = df["DATA"].apply(preprocess_text)

y = df["TOPIC"]

# TF-IDF 
tfidf_word = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000)

X_word = tfidf_word.fit_transform(df["DATA"])
X_char = tfidf_char.fit_transform(df["DATA"])
X = hstack([X_word, X_char])

# SPLIT 
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_pool, y_train, y_pool = train_test_split(
    X_train_full, y_train_full, test_size=0.3, stratify=y_train_full, random_state=42
)

start_time = time.time()

# BASE MODELS 
nb = train_nb(X_train, y_train)
lr = train_lr(X_train, y_train)
svm = train_svm(X_train, y_train)

end_time = time.time()

training_time = end_time - start_time

print(f"Training Time: {training_time:.2f} seconds")

print("\nNAIVE BAYES RESULTS:\n")
print(classification_report(y_test, nb.predict(X_test), zero_division=0))

print("\nLOGISTIC REGRESSION RESULTS:\n")
print(classification_report(y_test, lr.predict(X_test), zero_division=0))

print("\nSVM RESULTS:\n")
print(classification_report(y_test, svm.predict(X_test), zero_division=0))

# ACTIVE LEARNING 
lr_probs = lr.predict_proba(X_pool)
nb_pred_pool = nb.predict(X_pool)
lr_pred_pool = lr.predict(X_pool)

uncertainty = 1 - np.max(lr_probs, axis=1)
disagreement = (nb_pred_pool != lr_pred_pool)

uncertain_idx = np.argsort(uncertainty)[-5000:]
disagree_idx = np.where(disagreement)[0][:5000]

hard_idx = np.unique(np.concatenate([uncertain_idx, disagree_idx]))

X_hard = X_pool[hard_idx]
y_hard = y_pool.iloc[hard_idx]

X_train_new = vstack([X_train, X_hard])
y_train_new = np.concatenate([y_train, y_hard])

lr = train_lr(X_train_new, y_train_new)

print("\nLOGISTIC REGRESSION AFTER ACTIVE LEARNING:\n")
print(classification_report(y_test, lr.predict(X_test), zero_division=0))

# META MODEL 
nb_probs = nb.predict_proba(X_test)
lr_probs = lr.predict_proba(X_test)

nb_pred = nb.predict(X_test)
lr_pred = lr.predict(X_test)
svm_pred = svm.predict(X_test)

meta_X, meta_y = [], []

for i in range(len(nb_pred)):
    nb_conf = np.max(nb_probs[i])
    lr_conf = np.max(lr_probs[i])

    meta_X.append([
        nb_conf,
        lr_conf,
        int(nb_pred[i] == lr_pred[i]),
        int(lr_pred[i] == svm_pred[i])
    ])

    # choose better model
    if lr_pred[i] == y_test.iloc[i]:
        meta_y.append(1)
    else:
        meta_y.append(0)

meta = train_lr(np.array(meta_X), meta_y)

# FINAL PREDICTION
final_preds = []

for i in range(len(nb_pred)):
    votes = [nb_pred[i], lr_pred[i], svm_pred[i]]

    if votes.count(votes[0]) > 1:
        final_preds.append(votes[0])
    elif votes.count(votes[1]) > 1:
        final_preds.append(votes[1])
    else:
        nb_conf = np.max(nb_probs[i])
        lr_conf = np.max(lr_probs[i])

        choice = meta.predict([[
            nb_conf,
            lr_conf,
            int(nb_pred[i] == lr_pred[i]),
            int(lr_pred[i] == svm_pred[i])
        ]])[0]

        final_preds.append(nb_pred[i] if choice == 0 else lr_pred[i])

print("\nFINAL RESULTS:\n")
print(classification_report(y_test, final_preds, zero_division=0))

pickle.dump(nb, open("final_models/nb.pkl", "wb"))
pickle.dump(lr, open("final_models/lr.pkl", "wb"))
pickle.dump(svm, open("final_models/svm.pkl", "wb"))
pickle.dump(meta, open("final_models/meta.pkl", "wb"))
pickle.dump((tfidf_word, tfidf_char), open("final_models/vectorizer.pkl", "wb"))




