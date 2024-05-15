import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    st.title("Tennis Play Prediction")

    # Load data from CSV
    
    def load_data():
        data = pd.read_csv('C:/Users/MERLLIN YAZHINI/Desktop/Machine learning obser/wee4/tennisdata.csv')
        return data

    data = load_data()

    st.write("The first 5 values of data are:")
    st.write(data.head())

    # Obtain Train data and Train output
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert them to numbers
    le_outlook = LabelEncoder()
    X.Outlook = le_outlook.fit_transform(X.Outlook)

    le_Temperature = LabelEncoder()
    X.Temperature = le_Temperature.fit_transform(X.Temperature)

    le_Humidity = LabelEncoder()
    X.Humidity = le_Humidity.fit_transform(X.Humidity)

    le_Windy = LabelEncoder()
    X.Windy = le_Windy.fit_transform(X.Windy)

    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    accuracy = accuracy_score(classifier.predict(X_test), y_test)

    st.write(f"Accuracy is: {accuracy}")

if __name__ == "__main__":
    main()
