import joblib
import re

svm = joblib.load("model/svm_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

while True:
    text = input("\nMasukkan review (ketik 'exit'): ")
    if text.lower() == "exit":
        break

    clean = preprocess(text)
    vector = tfidf.transform([clean])
    pred = svm.predict(vector)[0]

    print("Sentimen:", "Positif" if pred == 1 else "Negatif")
