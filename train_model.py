import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine data
data = pd.concat([fake, true])
data = data[['text', 'label']]

# Split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tf, y_train)

# Evaluate
y_pred = model.predict(X_test_tf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
