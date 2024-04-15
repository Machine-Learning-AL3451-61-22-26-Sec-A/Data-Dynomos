import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_synthetic_data():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # Three classes
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
    df['target'] = y
    return df

def train_model(df):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    st.title("Decision Tree Classifier")
    st.write("This app demonstrates the working of a Decision Tree Classifier using synthetic data.")

    # Generate synthetic data
    df = generate_synthetic_data()

    # Display dataset
    st.subheader("Synthetic Dataset")
    st.write(df)

    # Train model
    model, X_test, y_test = train_model(df)

    # Evaluate model
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))

    # Classification of new data
    st.subheader("Classify New Data")
    feature_inputs = []
    for i in range(X_test.shape[1]):
        feature_input = st.number_input(f"Enter feature {i+1}:", step=0.1)
        feature_inputs.append(feature_input)
    if st.button("Predict"):
        new_data = [feature_inputs]
        prediction = model.predict(new_data)[0]
        st.write("Predicted Class:", prediction)

if __name__ == "__main__":
    main()
