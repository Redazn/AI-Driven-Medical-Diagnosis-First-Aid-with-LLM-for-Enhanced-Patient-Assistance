# AI-Driven-Medical-Diagnosis-First-Aid-with-LLM-for-Enhanced-Patient-Assistance


Medical AI adalah chatbot berbasis AI yang membantu memberikan edukasi dan analisis awal terhadap gejala medis yang dirasakan pasien. Chatbot ini menggunakan kombinasi model BioBERT, BioGPT, dan Gemini untuk mendeteksi kondisi, memberikan penjelasan, dan merekomendasikan konsultasi dengan spesialis terkait.


---

Fitur Utama

Ekstraksi entitas medis dengan BioBERT

Deteksi kondisi dan pencocokan gejala menggunakan database medis statis

Generasi jawaban medis awal menggunakan BioGPT

Penyempurnaan dan penyederhanaan bahasa medis menggunakan Gemini

Benchmarking sistem dengan test case gejala nyata

Antarmuka pengguna interaktif berbasis IPython widgets



---

Arsitektur

1. BioBERT: Untuk ekstraksi entitas dari pertanyaan medis


2. BioGPT: Untuk menghasilkan draft jawaban berdasarkan instruksi format medis


3. Gemini 1.5 Flash: Untuk menyederhanakan dan merapikan jawaban akhir


4. Database Medis: Format dictionary statis berisi gejala, obat, spesialis, dan kode ICD-10


5. Benchmark Tools: Evaluasi performa berbasis keyword matching dan exact match




---

Instalasi

pip install transformers torch evaluate google-generativeai ipywidgets pandas


---

Struktur Proyek

├── main.py
├── README.md
└── requirements.txt


---

Cara Menjalankan

python main.py


---

Hardware dan Kecepatan Inference

Perangkat uji : Google Colab (GPU Tesla T4/RTX A100 tergantung persediaan)

Kecepatan Inference (rata-rata):

BioBERT entity extraction: ~0.3s

BioGPT response generation (300 token): ~1.8s

Gemini post-processing: ~1.2s


Total Latensi Rata-Rata: 3.3s per pertanyaan



---

API Gemini

Pastikan untuk mengganti API_KEY_ANDA dengan API key dari Google Gemini di bagian konfigurasi:

genai.configure(api_key="API_KEY_ANDA")


---

Test Case dan Evaluasi

Sistem diuji dengan 2 kasus gejala umum:

Diabetes: "Saya sering haus dan penglihatan kabur"

Hipertensi: "Kepala saya sakit dan mimisan"


Hasil evaluasi mencakup:

Keyword Accuracy

Exact Match

Deteksi kata kunci gejala



---

Penanganan Error

Tiga kelas error utama:

SafetyFilterError: Prompt diblokir oleh sistem

ModelOverloadError: Keterbatasan memori GPU

MedicalChatError: Error umum lainnya dalam proses



---

Catatan Penggunaan

> Penting: MedicalAI bukan pengganti diagnosis medis profesional. Selalu konsultasikan kondisi Anda ke dokter.




---

Lisensi

Proyek ini open-source dan bebas digunakan untuk keperluan edukasi dan penelitian. Mohon sertakan atribusi apabila digunakan secara publik.


---

Kontributor


