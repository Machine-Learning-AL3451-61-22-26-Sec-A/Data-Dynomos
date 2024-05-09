import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Load data
msg = pd.read_csv('document.csv', names=['message', 'label'])

# Preprocess data
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

# Vectorize text
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)

# Display predictions
st.write("Predictions:")
for doc, p in zip(Xtest, pred):
    p = 'pos' if p == 1 else 'neg'
    st.write(f"{doc} -> {p}")

# Calculate metrics
accuracy = accuracy_score(ytest, pred)
recall = recall_score(ytest, pred)
precision = precision_score(ytest, pred)
conf_matrix = confusion_matrix(ytest, pred)

# Display metrics
st.write('Accuracy Metrics:')
st.write('Accuracy:', accuracy)
st.write('Recall:', recall)
st.write('Precision:', precision)
st.write('Confusion Matrix:', conf_matrix)
