import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string

def load_data(filepath):
    data = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    return data

def preprocess_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def predict_message(clf, vectorizer, message):
    message = message.lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    msg_vec = vectorizer.transform([message])
    pred = clf.predict(msg_vec)
    return "Spam" if pred[0] == 1 else "Not Spam"

def main():
    data = load_data('SMSSpamCollection')
    data['message'] = data['message'].apply(preprocess_text)
    data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

    X = data['message']
    y = data['label_num']

    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Test your own messages
    while True:
        msg = input("\nEnter an SMS message to classify (or type 'exit' to quit): ")
        if msg.lower() == 'exit':
            break
        result = predict_message(clf, vectorizer, msg)
        print(f"Prediction: {result}")

if __name__ == "__main__":
    main()
