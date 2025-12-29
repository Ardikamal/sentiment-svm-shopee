import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD DATASET
# =========================
DATA_PATH = "data/shopee_reviews.csv"  # pastikan file ada di folder data/

df = pd.read_csv(DATA_PATH)

print("Kolom dataset:", df.columns.tolist())

# =========================
# PILIH KOLOM
# =========================
TEXT_COL = "content"
SCORE_COL = "score"

# =========================
# BUAT LABEL SENTIMEN
# =========================
def label_sentiment(score):
    if score <= 2:
        return 0  # Negatif
    elif score >= 4:
        return 1  # Positif
    else:
        return None  # Netral dibuang

df["label"] = df[SCORE_COL].apply(label_sentiment)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\nDistribusi Label:")
print(df["label"].value_counts())

# =========================
# PREPROCESSING TEXT
# =========================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nBefore cleaning:")
print(df[TEXT_COL].iloc[0])

print("\nAfter cleaning:")
print(preprocess(df[TEXT_COL].iloc[0]))

df["clean_text"] = df[TEXT_COL].apply(preprocess)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# SVM MODEL
# =========================
svm = SVC(kernel="linear", C=1)
svm.fit(X_train_tfidf, y_train)

y_pred = svm.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negatif", "Positif"],
    yticklabels=["Negatif", "Positif"]
)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - SVM Shopee")
plt.show()

# =========================
# SAVE MODEL
# =========================
joblib.dump(svm, "model/svm_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")

print("\nModel & TF-IDF berhasil disimpan!")
