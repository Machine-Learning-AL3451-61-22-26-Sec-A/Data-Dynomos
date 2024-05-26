import streamlit as st
import pandas as pd
import numpy as np

# Title of the app
st.title("k-Nearest Neighbour (k-NN) Classification of Iris Dataset")

# Manually load Iris dataset
def load_iris_dataset():
    data = {
        'sepal length (cm)': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                              5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1,
                              5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7,
                              4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1,
                              5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0],
        'sepal width (cm)': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                             3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8,
                             3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2,
                             3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3.0, 3.4,
                             3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3],
        'petal length (cm)': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                              1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5,
                              1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6,
                              1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3, 1.5,
                              1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4],
        'petal width (cm)': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
                             0.2, 0.2, 0.1, 0.1, 0.4, 0.4, 0.3, 0.3, 0.4, 0.3,
                             0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2,
                             0.2, 0.4, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.3,
                             0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2],
        'target': [0]*50 + [1]*50 + [2]*50
    }
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_names = np.array(['setosa', 'versicolor', 'virginica'])
    return pd.DataFrame(data), target_names

# Euclidean distance calculation
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# k-NN algorithm implementation
def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        labels = [label for _, label in neighbors]
        prediction = max(set(labels), key=labels.count)
        y_pred.append(prediction)
    return np.array(y_pred)

# Function to split dataset
def train_test_split(data, test_size=0.3):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# Load and display Iris dataset
data, target_names = load_iris_dataset()
st.subheader("Iris Dataset")
st.write(data)

# User input for k
k = st.slider("Select the number of neighbors (k):", min_value=1, max_value=20, value=5)

# Split the data into training and test sets
train_data, test_data = train_test_split(data)
X_train = train_data.drop(columns=['target']).values
y_train = train_data['target'].values
X_test = test_data.drop(columns=['target']).values
y_test = test_data['target'].values

# Make predictions
y_pred = knn_predict(X_train, y_train, X_test, k)

# Calculate accuracy
accuracy = np.sum(y_test == y_pred) / len(y_test)

# Display accuracy
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Show correct and wrong predictions
st.subheader("Correct and Wrong Predictions")
correct_predictions = []
wrong_predictions = []

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct_predictions.append((X_test[i], target_names[y_test[i]], target_names[y_pred[i]]))
    else:
        wrong_predictions.append((X_test[i], target_names[y_test[i]], target_names[y_pred[i]]))

# Display correct predictions
st.subheader("Correct Predictions")
for x, true_label, pred_label in correct_predictions:
    st.write(f"Data: {x}, True Label: {true_label}, Predicted Label: {pred_label}")

# Display wrong predictions
st.subheader("Wrong Predictions")
for x, true_label, pred_label in wrong_predictions:
    st.write(f"Data: {x}, True Label: {true_label}, Predicted Label: {pred_label}")

