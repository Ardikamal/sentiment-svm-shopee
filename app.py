import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# =====================================================
# CUSTOM CSS (UI UPGRADE)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.main {
    padding-top: 1rem;
}
h1, h2, h3 {
    color: #1f2937;
}
.card {
    background-color: white;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="card">
    <h2 style="text-align:center;">💬 Analisis Sentimen Ulasan Pengguna</h2>
    <p style="text-align:center;">
        Menggunakan <b>TF-IDF</b> dan <b>Support Vector Machine (SVM)</b>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<b>📌 Alur Penggunaan Aplikasi</b><br>
1️⃣ Upload dataset ulasan (CSV / XLSX)<br>
2️⃣ Pilih kolom teks dan skor<br>
3️⃣ Train model sentimen<br>
4️⃣ Uji sentimen secara interaktif
</div>
""", unsafe_allow_html=True)

# =====================================================
# FUNCTIONS
# =====================================================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def label_sentiment(score):
    if pd.isna(score):
        return None
    if score <= 2:
        return 0
    elif score >= 4:
        return 1
    else:
        return None

# =====================================================
# SESSION STATE
# =====================================================
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# =====================================================
# UPLOAD DATASET
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Dataset")

uploaded_file = st.file_uploader(
    "Unggah dataset ulasan dalam format CSV atau XLSX",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset berhasil diupload")

    with st.expander("🔍 Lihat contoh dataset"):
        st.dataframe(df.head())

    st.info("""
    **Ketentuan Dataset**
    - Memiliki kolom teks ulasan
    - Memiliki kolom skor (1–5)
    - Skor 1–2 → Negatif  
    - Skor 4–5 → Positif  
    - Skor 3 → Tidak digunakan
    """)

    # =====================================================
    # KONFIGURASI DATASET
    # =====================================================
    st.subheader("⚙️ Konfigurasi Dataset")

    text_col = st.selectbox("Kolom teks ulasan", df.columns)
    score_col = st.selectbox("Kolom skor", df.columns)

    # =====================================================
    # TRAIN MODEL
    # =====================================================
    if st.button("🚀 Train Model"):
        with st.spinner("⏳ Melatih model sentimen..."):
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
            df["label"] = df[score_col].apply(label_sentiment)
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)

            df["clean_text"] = df[text_col].apply(preprocess)

            X_train, X_test, y_train, y_test = train_test_split(
                df["clean_text"],
                df["label"],
                test_size=0.2,
                random_state=42,
                stratify=df["label"]
            )

            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)

            svm = SVC(kernel="linear", C=1)
            svm.fit(X_train_tfidf, y_train)

            y_pred = svm.predict(X_test_tfidf)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"🎯 Training selesai | Akurasi: {acc:.2f}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negatif", "Positif"],
                yticklabels=["Negatif", "Positif"],
                ax=ax
            )
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            joblib.dump(svm, "svm_model.pkl")
            joblib.dump(tfidf, "tfidf.pkl")

            st.session_state.model_trained = True
            st.session_state.svm = svm
            st.session_state.tfidf = tfidf

            st.success("💾 Model berhasil disimpan dan siap digunakan")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDIKSI MANUAL
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("✍️ Uji Sentimen Ulasan")

if not st.session_state.model_trained:
    st.warning("⚠️ Silakan train model terlebih dahulu.")
else:
    user_text = st.text_area(
        "Masukkan ulasan pengguna",
        placeholder="Contoh: aplikasi ini sangat membantu dan mudah digunakan"
    )

    if st.button("🔍 Prediksi Sentimen"):
        clean = preprocess(user_text)
        vec = st.session_state.tfidf.transform([clean])
        pred = st.session_state.svm.predict(vec)[0]

        if pred == 1:
            st.success("😊 Sentimen POSITIF")
        else:
            st.error("😠 Sentimen NEGATIF")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
Aplikasi Analisis Sentimen • SVM & TF-IDF • Streamlit
</div>
""", unsafe_allow_html=True)
