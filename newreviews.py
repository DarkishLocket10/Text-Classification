import pandas as pd
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Extract the data file from the tar file
tar = tarfile.open('aclImdb_v1.tar.gz')
data_file = tar.extractfile('aclImdb/train/labeledBow.feat')

# Load the data file into a pandas dataframe
df = pd.read_csv(data_file,
                 header=0,
                 sep='\t',
                 quotechar='"')


# Check if the 'sentiment' column exists in the dataframe
if 'sentiment' in df:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.33, random_state=42)
else:
    print("The 'sentiment' column does not exist in the dataframe.")
    
# Create a CountVectorizer to turn the texts into numerical feature vectors
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# Fit a Logistic Regression model to the training data
model = LogisticRegression()
model.fit(train_vectors, y_train)

# Calculate the model's accuracy on the test set
accuracy = model.score(test_vectors, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Define a function to predict the sentiment of a given string
def predict_sentiment(text):
    # Transform the text into a numerical feature vector using the trained vectorizer
    vector = vectorizer.transform([text])
    # Use the model to make a prediction
    prediction = model.predict(vector)[0]
    # Return the predicted label (1 for positive, 0 for negative)
    return "positive" if prediction == 1 else "negative"
