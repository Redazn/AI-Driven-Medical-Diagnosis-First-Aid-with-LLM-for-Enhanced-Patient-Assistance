
# **AI-Driven Medical Diagnosis & First Aid with LLM for Enhanced Patient Assistance**

## **Deskripsi Proyek**
Proyek ini adalah **chatbot berbasis AI** yang dirancang untuk membantu diagnosis medis awal dan memberikan saran pertolongan pertama. Dengan menggabungkan kekuatan **Large Language Models (LLMs)** seperti BioGPT dan BioBERT, sistem ini memperkaya pengalaman pengguna melalui:
- **Retrieval-Augmented Generation (RAG)** untuk pencarian jawaban akurat dari dataset medis.
- **Confidence Scoring System** untuk memberikan kepercayaan pada keluaran sistem.
- **Refinement** menggunakan Gemini AI untuk meningkatkan kualitas respons.

> **Catatan Penting**:
> Aplikasi ini bertujuan untuk memberikan informasi medis yang edukatif dan **tidak menggantikan diagnosis atau konsultasi medis profesional, tidak ada garansi jika anda menggunakan secara komersial!**.

---

## **Fitur Utama**
### **1. Integrasi LLMs**
- **BioBERT**: Digunakan untuk ekstraksi entitas medis dari input pengguna.
- **BioGPT**: Digunakan untuk menghasilkan teks berbasis gejala medis dan konteks.
- **Gemini AI**: Untuk penyempurnaan jawaban sehingga lebih mudah dipahami oleh pengguna.

### **2. Retrieval-Augmented Generation (RAG)**
- Menerapkan **FAISS** sebagai mesin pencari berbasis embedding untuk mengambil jawaban dari dataset medis.
- Dataset yang digunakan meliputi **MedQuad**, **Medical Meadow Flashcards**, dan **Medical Meadow WikiDoc**.

### **3. Confidence Scoring System**
- Sistem penilaian kepercayaan menghitung skor berdasarkan:
  - **Semantic similarity** antara konteks dan jawaban.
  - **Completeness** dari respons terhadap pertanyaan.
  - **Medical terminology** yang digunakan dalam jawaban.
  - **Certainty** dan analisis ketidakpastian dalam respons.
  - **Length appropriateness** untuk memastikan panjang jawaban relevan.
  - **Structure assessment** untuk memastikan format jawaban terstruktur.

### **4. User Interface (UI)**
- **Berbasis Jupyter Notebook** menggunakan `ipywidgets` untuk input pertanyaan dan visualisasi metrik kepercayaan.
- **Confidence Gauge**: Menampilkan tingkat kepercayaan jawaban secara visual.
- **Polar Chart Visualization**: Memvisualisasikan metrik kepercayaan menggunakan `plotly`.

### **5. Benchmarking**
- Fungsi benchmarking untuk menguji performa sistem terhadap berbagai kasus uji medis.

---

## **Struktur Proyek**
- `app.py`: File utama yang mencakup semua logika sistem, termasuk pemrosesan input, model loading, RAG, confidence scoring, dan UI.
- `requirements.txt`: Daftar dependensi Python yang diperlukan untuk menjalankan proyek.

---

## **Teknologi yang Digunakan**
- **Bahasa Pemrograman**: Python
- **Framework dan Library**:
  - `transformers`, `torch`: Untuk integrasi LLMs.
  - `sentence-transformers`, `faiss`: Untuk RAG dan embedding.
  - `ipywidgets`, `plotly`: Untuk antarmuka pengguna.
  - `google-generativeai`: Untuk penyempurnaan jawaban dengan Gemini AI.

---

## **Cara Instalasi**
### **1. Persyaratan Sistem**
- Python 3.8 atau lebih baru
- GPU dengan CUDA (opsional, untuk akselerasi pemrosesan)
- Jupyter Notebook atau JupyterLab

### **2. Instalasi Dependensi**
1. Clone repositori:
   ```bash
   git clone https://github.com/Redazn/AI-Driven-Medical-Diagnosis-First-Aid-with-LLM-for-Enhanced-Patient-Assistance.git
   cd AI-Driven-Medical-Diagnosis-First-Aid-with-LLM-for-Enhanced-Patient-Assistance
   ```
2. Instal semua dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Ganti API key Anda: ```genai.configure(api_key="GANTI_API_KEY_ANDA")  # Ganti dengan API key```

---

## **Cara Penggunaan**
### **1. Menjalankan Aplikasi**
1. Buka Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Jalankan file `app.py`.

### **2. Menggunakan Antarmuka**
- Masukkan pertanyaan medis ke dalam kotak teks.
- Klik **"Dapatkan Jawaban"** untuk menerima respons dari sistem.
- Visualisasi tingkat kepercayaan jawaban akan ditampilkan melalui **Confidence Gauge** dan **Polar Chart**.

### **3. Menjalankan Benchmark**
- Fungsi `run_benchmark()` akan otomatis dijalankan untuk menguji performa sistem terhadap kasus uji.

---

## **Batasan Sistem**
1. **Bukan Pengganti Diagnosis Medis**:
   - Aplikasi ini tidak dimaksudkan untuk menggantikan konsultasi dokter atau diagnosis profesional.
2. **Ketergantungan pada Dataset**:
   - Akurasi jawaban tergantung pada data yang tersedia di dataset medis.
3. **Keterbatasan GPU**:
   - Penggunaan GPU disarankan untuk performa optimal, tetapi fallback ke CPU tersedia.
4. **Keamanan Data**:
   - Belum ada enkripsi untuk input pengguna; tidak cocok untuk data pribadi yang sensitif.

---

## **Pengembangan Selanjutnya**
1. **Deployment Berbasis Web**:
   - Menggunakan framework seperti Streamlit atau FastAPI untuk membuat aplikasi lebih mudah diakses.
2. **Ekspansi Dataset**:
   - Menambahkan lebih banyak dataset untuk memperluas cakupan kondisi medis.
3. **Optimasi Confidence Scoring**:
   - Mengintegrasikan model evaluasi otomatis untuk meningkatkan akurasi skor.
4. **Keamanan Data**:
   - Implementasi enkripsi untuk melindungi data pengguna.

---

## **Lisensi**
Proyek ini berlisensi yang memungkinkan penggunaan, modifikasi dan distribusi bebas, termasuk untuk keperluan edukasi dan penelitian, selama syarat Lisensi terpenuhi [Apache 2.0 License](LICENSE).

---

## **Kontribusi**
Kontribusi sangat dihargai! Jika Anda ingin berkontribusi:
1. Fork repositori ini.
2. Buat branch baru untuk fitur atau perbaikan Anda.
3. Kirim Pull Request (PR) dengan deskripsi yang jelas.

---

## **Kontak**
Jika Anda memiliki pertanyaan atau saran, silakan hubungi melalui [issues](https://github.com/Redazn/AI-Driven-Medical-Diagnosis-First-Aid-with-LLM-for-Enhanced-Patient-Assistance/issues).

---
