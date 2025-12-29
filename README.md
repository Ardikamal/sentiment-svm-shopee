
# 💬 Sentiment Analysis Ulasan Pengguna dengan SVM

Aplikasi ini merupakan sistem **Analisis Sentimen Teks** yang mengklasifikasikan ulasan pengguna ke dalam **sentimen Positif dan Negatif** menggunakan metode **Support Vector Machine (SVM)** dan **TF-IDF**.  
Sistem dikembangkan menggunakan **Python** dan **Streamlit**, serta mendukung **unggah dataset CSV/XLSX** secara fleksibel.

---

## 📌 Fitur Utama
- Upload dataset ulasan (CSV / XLSX)
- Preprocessing teks otomatis (cleaning & normalisasi)
- Ekstraksi fitur menggunakan TF-IDF
- Klasifikasi sentimen menggunakan SVM (Linear & RBF)
- Evaluasi model (Akurasi & Confusion Matrix)
- Visualisasi WordCloud Positif & Negatif
- Prediksi sentimen secara interaktif
- Tampilan responsif (Desktop & Mobile)

---

## 🧠 Metodologi
Tahapan analisis sentimen yang diterapkan pada sistem ini meliputi:
1. **Load Dataset**
2. **Text Preprocessing**
   - Case folding
   - Cleaning (hapus angka, simbol, dan URL)
3. **Feature Extraction**
   - TF-IDF (Unigram & Bigram)
4. **Modeling**
   - Support Vector Machine (Kernel Linear & RBF)
5. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
6. **Deployment**
   - Web App berbasis Streamlit

---

## 📂 Struktur Project
```text
sentiment-svm/
│
├── app.py                 # Streamlit Web Application
├── train_model.py         # Training & evaluasi model
├── predict.py             # Prediksi via terminal
│
├── data/
│   └── shopee_reviews.csv # Dataset
│
├── models/
│   ├── svm_model.pkl      # Model SVM
│   └── tfidf.pkl          # TF-IDF Vectorizer
│
├── requirements.txt
└── README.md
````

---

## 📊 Dataset

Dataset berupa ulasan pengguna e-commerce Shopee yang memiliki kolom:

* `content` : teks ulasan
* `score`   : rating (1–5)

Label sentimen dibentuk sebagai berikut:

* Skor ≤ 2 → **Negatif**
* Skor ≥ 4 → **Positif**
* Skor 3 → Tidak digunakan

Dataset dapat berasal dari Kaggle atau sumber lain selama memiliki struktur serupa.

---

## ⚙️ Instalasi & Menjalankan Program

### 1️⃣ Clone Repository

```bash
git clone https://github.com/username/sentiment-svm.git
cd sentiment-svm
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Jalankan Training Model

```bash
python train_model.py
```

### 4️⃣ Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi dapat diakses melalui browser pada:

```
http://localhost:8501
```

---

## 🧪 Contoh Prediksi

Input:

```
Aplikasi ini sangat membantu dan mudah digunakan
```

Output:

```
Sentimen: Positif
```

Input:

```
Aplikasinya buruk dan sering error
```

Output:

```
Sentimen: Negatif
```

---

## 📈 Evaluasi Model

Evaluasi dilakukan menggunakan **Confusion Matrix** dan **Accuracy Score**.
Hasil eksperimen menunjukkan bahwa **SVM dengan kernel Linear** memberikan performa lebih baik dibandingkan kernel RBF pada data teks berdimensi tinggi.

---

## 🌐 Deployment

Aplikasi ini dapat dideploy menggunakan:

* Streamlit Cloud
* Docker
* Local Server

Tampilan aplikasi dirancang responsif dan dapat diakses melalui perangkat desktop maupun mobile.

---

## 📄 Lisensi

Project ini dikembangkan untuk keperluan akademik dan pembelajaran.
Silakan gunakan dan kembangkan sesuai kebutuhan.

---

## 👨‍💻 Author

**Ardi Kamal Karima**
Mahasiswa Informatika

---

