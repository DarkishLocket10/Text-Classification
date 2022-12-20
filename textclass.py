import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Example training data
train_texts = [
    "This movie was excellent",
    "The acting was really good",
    "I liked the story and the characters",
    "The movie was very boring",
    "I did not enjoy the movie",
    "The movie was terrible",
]

# Example labels for the training data (1 for positive, 0 for negative)
train_labels = [1, 1, 1, 0, 0, 0]

# Create a CountVectorizer to turn the texts into numerical feature vectors
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_texts)

# Fit a Logistic Regression model to the training data
model = LogisticRegression()
model.fit(train_vectors, train_labels)

# Define a function to predict the sentiment of a given string
def predict_sentiment(text):
    # Transform the text into a numerical feature vector using the trained vectorizer
    vector = vectorizer.transform([text])
    # Use the model to make a prediction
    prediction = model.predict(vector)[0]
    # Return the predicted label (1 for positive, 0 for negative)
    return "positive" if prediction == 1 else "negative"

# Test the function with some example strings
test_texts = [
    "This movie was terrible",
    "I really enjoyed the movie",
    "The acting was mediocre but the story was good",
]

for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f'"{text}" is {sentiment}')
