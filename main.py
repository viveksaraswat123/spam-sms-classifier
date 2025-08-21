import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@st.cache_resource
def load_and_train_model():
    data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
    data['message'] = data['message'].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
    data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

    X = data['message']
    y = data['label_num']

    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return clf, vectorizer, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

def preprocess_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def predict_message(clf, vectorizer, message):
    message = preprocess_text(message)
    msg_vec = vectorizer.transform([message])
    pred = clf.predict(msg_vec)
    return "Spam" if pred[0] == 1 else "Not Spam"

def main():
    st.title("📱 SMS Spam Classifier")

    clf, vectorizer, accuracy, class_report, conf_matrix = load_and_train_model()

    st.write(f"**Model Accuracy:** {accuracy:.2%}")

    if st.checkbox("Show Classification Report and Confusion Matrix"):
        st.text(class_report)
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

    user_input = st.text_area("Enter SMS message to classify:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.error("Please enter a message to classify.")
        else:
            result = predict_message(clf, vectorizer, user_input)
            st.success(f"Prediction: **{result}**")

if __name__ == "__main__":
    main()
