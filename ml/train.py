import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test), target_names=["ham", "spam"]))

joblib.dump(model, "ml/spam_model.joblib")
print("Model saved to ml/spam_model.joblib")