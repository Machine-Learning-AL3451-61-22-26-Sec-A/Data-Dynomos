import pandas as pd
import streamlit as st

# Load data
msg = pd.read_csv('document.csv', names=['message', 'label'])

# Preprocess data
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum

# Split data
def train_test_split(X, y, test_size=0.2, random_state=None):
    n_samples = len(X)
    test_samples = int(test_size * n_samples)
    if random_state:
        import random
        random.seed(random_state)
        indices = random.sample(range(n_samples), test_samples)
    else:
        import numpy as np
        indices = np.random.choice(n_samples, test_samples, replace=False)
    X_train = X.drop(indices)
    X_test = X.iloc[indices]
    y_train = y.drop(indices)
    y_test = y.iloc[indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Vectorize text
class CountVectorizer:
    def __init__(self):
        self.vocab = {}
    
    def fit_transform(self, X):
        for text in X:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        X_dm = []
        for text in X:
            doc_vector = [0] * len(self.vocab)
            for word in text.split():
                if word in self.vocab:
                    doc_vector[self.vocab[word]] += 1
            X_dm.append(doc_vector)
        return X_dm

    def transform(self, X):
        X_dm = []
        for text in X:
            doc_vector = [0] * len(self.vocab)
            for word in text.split():
                if word in self.vocab:
                    doc_vector[self.vocab[word]] += 1
            X_dm.append(doc_vector)
        return X_dm

count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(X_train)
Xtest_dm = count_v.transform(X_test)

# Naive Bayes classifier
class MultinomialNB:
    def __init__(self):
        self.p_pos = None
        self.p_neg = None
    
    def fit(self, X, y):
        pos_count = 0
        neg_count = 0
        pos_freq = [0] * len(X[0])
        neg_freq = [0] * len(X[0])
        for i in range(len(X)):
            if y.iloc[i] == 1:
                pos_count += 1
                for j in range(len(X[i])):
                    pos_freq[j] += X[i][j]
            else:
                neg_count += 1
                for j in range(len(X[i])):
                    neg_freq[j] += X[i][j]
        self.p_pos = [freq / sum(pos_freq) for freq in pos_freq]
        self.p_neg = [freq / sum(neg_freq) for freq in neg_freq]
    
    def predict(self, X):
        preds = []
        for doc in X:
            pos_prob = 1
            neg_prob = 1
            for i in range(len(doc)):
                if doc[i] > 0:
                    pos_prob *= self.p_pos[i] ** doc[i]
                    neg_prob *= self.p_neg[i] ** doc[i]
            pred = 1 if pos_prob > neg_prob else 0
            preds.append(pred)
        return preds

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(Xtrain_dm, y_train)
pred = clf.predict(Xtest_dm)

# Display predictions
st.write("Predictions:")
for doc, p in zip(X_test, pred):
    p = 'pos' if p == 1 else 'neg'
    st.write(f"{doc} -> {p}")

# Calculate metrics
def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred):
    cm = [[0, 0], [0, 0]]
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

def precision_score(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

# Display metrics
accuracy = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)

st.write('Accuracy Metrics:')
st.write('Accuracy:', accuracy)
st.write('Recall:', recall)
st.write('Precision:', precision)
st.write('Confusion Matrix:', conf_matrix)
