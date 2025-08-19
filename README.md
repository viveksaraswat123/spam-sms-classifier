**Spam SMS Classifier**

This project is a simple yet effective SMS spam detection system built using Python and machine learning. It uses the popular Naive Bayes algorithm to classify SMS messages as Spam or Ham (not spam).

The model is trained on the publicly available SMS Spam Collection Dataset from the UCI Machine Learning Repository, which contains thousands of labeled SMS messages.

**Features**

Loads and preprocesses raw SMS data (lowercasing, punctuation removal).

Converts text messages into numerical features using CountVectorizer.

Trains a Multinomial Naive Bayes classifier for spam detection.

Evaluates model performance with accuracy, classification report, and confusion matrix.

Interactive testing: enter any SMS message to see if it’s spam or ham.

Easy to run and extend for anyone learning machine learning or natural language processing.

**How to Run**

1. Download the dataset from GitHub Repo 

2. Extract the file SMSSpamCollection into the project directory.

3. Install required libraries: pip install pandas scikit-learn

4. Run the program: python spam_classifier.py


Enter your own SMS messages to test the classifier.

**What I Learned**

Basic natural language processing (text preprocessing, vectorization).

Implementing and evaluating a Naive Bayes classification model.

Using Python libraries like Pandas and scikit-learn.

Building an interactive command-line application.




