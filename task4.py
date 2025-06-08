import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load the dataset
# Example dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# For this example, create a small dataset manually:
data = {
    'label': ['ham', 'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
    'message': [
        'Hey, how are you?',
        'Win money now!!!',
        'Are we still meeting at 5pm?',
        'I will call you later',
        'Congratulations! You won a prize',
        'See you tomorrow at the meeting',
        'Get a free gift card now',
        'Let’s catch up soon!',
        'Free entry in 2 a weekly competition',
        'What’s up?'
    ]
}

df = pd.DataFrame(data)

# 2️⃣ Preprocess the data
X = df['message']
y = df['label']

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# 3️⃣ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4️⃣ Build the model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5️⃣ Make predictions
y_pred = model.predict(X_test)

# 6️⃣ Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
