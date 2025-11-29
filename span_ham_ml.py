# span_ham_ml_interactive.py

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------------------
# 1. NLTK Downloads
# ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------
# 2. Load Dataset
# ---------------------------
try:
    data = pd.read_csv("data.txt", sep="|")
    if 'message' not in data.columns or 'label' not in data.columns:
        raise ValueError
except:
    data = pd.read_csv("data.txt", sep="|", names=["label", "message"])

# Remove missing messages
data.dropna(subset=['message'], inplace=True)

# ---------------------------
# 3. Preprocessing Function
# ---------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    if not words:  # Keep at least some words
        words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

data['cleaned'] = data['message'].apply(preprocess_text)

# ---------------------------
# 4. Feature Extraction
# ---------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned'])
y = data['label']

# ---------------------------
# 5. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 6. Train Naive Bayes Model
# ---------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------------------
# 7. Evaluate on Test Set
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 8. Interactive Prediction
# ---------------------------
def predict_email(text):
    cleaned_text = preprocess_text(text)
    vect_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vect_text)
    return prediction[0]

print("\n--- Spam Detection Interactive Mode ---")
while True:
    user_input = input("Enter an email to check (or type 'exit' to quit):\n")
    if user_input.lower() == 'exit':
        print("Exiting program.")
        break
    result = predict_email(user_input)
    print(f"Prediction: {result}\n")
