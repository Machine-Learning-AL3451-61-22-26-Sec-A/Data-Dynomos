import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names

def train_model(df):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    st.title("Decision Tree Classifier")
    st.write("This app demonstrates the working of a Decision Tree Classifier using the ID3 algorithm.")

    # Load data
    df, target_names = load_data()

    # Display dataset
    st.subheader("Dataset")
    st.write(df)

    # Train model
    model, X_test, y_test = train_model(df)

    # Evaluate model
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred, target_names=target_names))

    # Classification of new data
    st.subheader("Classify New Data")
    sepal_length = st.number_input("Enter sepal length:")
    sepal_width = st.number_input("Enter sepal width:")
    petal_length = st.number_input("Enter petal length:")
    petal_width = st.number_input("Enter petal width:")
    if st.button("Predict"):
        new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(new_data)[0]
        st.write("Predicted Class:", target_names[prediction])

if __name__ == "__main__":
    main()
