from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_nb(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    return model

def train_lr(X, y):
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X, y)
    return model

def train_svm(X, y):
    model = LinearSVC()
    model.fit(X, y)
    return model