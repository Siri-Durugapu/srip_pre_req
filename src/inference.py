import pickle
from scipy.sparse import hstack
from utils import preprocess_text

nb = pickle.load(open("final_models/nb.pkl","rb"))
lr = pickle.load(open("final_models/lr.pkl","rb"))
svm = pickle.load(open("final_models/svm.pkl","rb"))
meta = pickle.load(open("final_models/meta.pkl","rb"))
tfidf_word, tfidf_char = pickle.load(open("final_models/vectorizer.pkl","rb"))

def predict(text):
    text_clean = preprocess_text(text)

    X_word = tfidf_word.transform([text_clean])
    X_char = tfidf_char.transform([text_clean])
    X = hstack([X_word, X_char])

    nb_pred = nb.predict(X)[0]
    lr_pred = lr.predict(X)[0]
    svm_pred = svm.predict(X)[0]

    nb_prob = nb.predict_proba(X)[0]
    lr_prob = lr.predict_proba(X)[0]

    nb_conf = max(nb_prob)
    lr_conf = max(lr_prob)

    votes = [nb_pred, lr_pred, svm_pred]

    if votes.count(votes[0]) > 1:
        return votes[0]
    elif votes.count(votes[1]) > 1:
        return votes[1]

    meta_features = [[
        nb_conf,
        lr_conf,
        int(nb_pred == lr_pred),
        int(lr_pred == svm_pred)
    ]]

    choice = meta.predict(meta_features)[0]

    return nb_pred if choice == 0 else lr_pred

if __name__ == "__main__":
    text = input("Enter text: ")
    print("Predicted Topic:", predict(text))