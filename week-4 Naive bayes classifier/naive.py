import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    st.title("Tennis Play Prediction")

    # Create a DataFrame
    data = {
        'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
        'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
        'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'PlayTennis': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    df = pd.DataFrame(data)

    st.write("The first 5 values of data are:")
    st.write(df.head())

    # Obtain Train data and Train output
    X = df.iloc[:,:-1]
    st.write("\nThe First 5 values of train data are:\n", X.head())

    y = df.iloc[:,-1]
    st.write("\nThe first 5 values of Train output are:\n", y.head())

    # Convert them to numbers 
    le_outlook = LabelEncoder()
    X.Outlook = le_outlook.fit_transform(X.Outlook)

    le_Temperature = LabelEncoder()
    X.Temperature = le_Temperature.fit_transform(X.Temperature)

    le_Humidity = LabelEncoder()
    X.Humidity = le_Humidity.fit_transform(X.Humidity)

    st.write("\nNow the Train data is :\n",X.head())

    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)
    st.write("\nNow the Train output is\n",y)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

    classifier = GaussianNB()
    classifier.fit(X_train,y_train)

    st.write("Accuracy is:",accuracy_score(classifier.predict(X_test),y_test))

if __name__ == "__main__":
    main()
