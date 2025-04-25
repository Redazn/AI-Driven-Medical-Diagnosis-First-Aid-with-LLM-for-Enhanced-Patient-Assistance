# AI-Driven-Medical-Diagnosis-First-Aid-with-LLM-for-Enhanced-Patient-Assistance


Project ini adalah chatbot berbasis AI yang membantu memberikan edukasi dan analisis awal terhadap gejala medis yang dirasakan pasien. Chatbot ini menggunakan kombinasi model BioBERT, BioGPT, dan Gemini untuk mendeteksi kondisi, memberikan penjelasan, dan merekomendasikan konsultasi dengan spesialis terkait.


---

Fitur Utama

Ekstraksi entitas medis dengan BioBERT

Deteksi kondisi dan pencocokan gejala menggunakan database medis statis

Generasi jawaban medis awal menggunakan BioGPT

Penyempurnaan dan penyederhanaan bahasa medis menggunakan Gemini

Antarmuka pengguna interaktif berbasis IPython widgets



---

Arsitektur

1. BioBERT: Untuk ekstraksi entitas dari pertanyaan medis


2. BioGPT: Untuk menghasilkan draft jawaban berdasarkan instruksi format medis


3. Gemini 1.5 Flash: Untuk menyederhanakan dan merapikan jawaban akhir


4. Database Medis: 15 Format dictionary statis berisi gejala, obat, spesialis, dan kode ICD-10 (bisa ditambahkan manual, untuk kebutuhan spesifik)


5. Benchmark Tools: Evaluasi performa berbasis keyword matching dan exact match




---

Instalasi

pip install transformers torch evaluate google-generativeai ipywidgets pandas


---

Struktur Proyek

 main.py
 
 README.md

 requirements.txt


---

Cara Menjalankan

python main.py


---

Hardware dan Kecepatan Inference

Perangkat uji : Google Colab (GPU Tesla T4 with ram 16GB (GDDR6))

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

Sistem diuji dengan 10 kasus gejala umum:


Hasil evaluasi mencakup:

=Recall sensivity, Precision, Specificity, Threshold

Keyword Accuracy

= Moderade, akurasi (70%-80%) tapi risk underdiagnosis untuk kondisi serius.

Exact Match

= Perlu banyak Penyempitan DDx dengan skor klinis, Threshold terlalu konservatif untuk gejala beresiko tinggi

Deteksi kata kunci gejala

= Precision, False positif tinggi. Recall, False negatif tinggi. Threshold, tidak optimal.

---

Contoh Kinerja

Pasien: badan saya sering nyeri, kadang tulang tuh sering kaku di saat bangun tidur 

🤖 Asisten Medis (Waktu respon: 3.56s):

        ### Diagnosis Kemungkinan
Nyeri tubuh dan kekakuan sendi, terutama di pagi hari, bisa disebabkan beberapa hal. Bayangkan tubuhmu seperti mobil tua yang perlu pelumas. Bisa jadi "pelumas" sendimu (cairan sinovial) berkurang, sehingga pergerakan terasa kaku dan nyeri. Ini bisa terjadi karena usia, kurang gerak, atau bahkan kondisi seperti arthritis (radang sendi). Atau mungkin ada otot yang tegang karena aktivitas atau posisi tidur yang salah. 

💊 Rekomendasi Obat: Parasetamol (Paracetamol): Obat pereda nyeri yang dijual bebas. Ikuti petunjuk dosis pada kemasan. Jangan melebihi dosis maksimal harian.Ibuprofen (Brufen, dll.): Obat anti-inflamasi nonsteroid (OAINS) yang juga dijual bebas, membantu mengurangi peradangan dan nyeri. Ikuti petunjuk dosis pada kemasan. Jangan digunakan jika Anda memiliki riwayat maag. 

🏥 Konsultasi Spesialis: Jika nyeri dan kekakuan terus berlanjut atau memburuk, sebaiknya konsultasikan ke dokter umum atau spesialis Reumatologi (ahli penyakit sendi dan rematik) atau Ortopedi (ahli penyakit tulang dan sendi). 

⚠️ Peringatan: Hasil ini bukan diagnosis pasti. Segera hubungi dokter jika nyeri sangat hebat, disertai demam, bengkak di sendi yang signifikan, atau keluhan lain yang mengkhawatirkan. Jangan mengonsumsi obat-obatan OTC secara berlebihan tanpa berkonsultasi dengan dokter. 

📌 Kode ICD-10: (Tidak dapat diberikan tanpa pemeriksaan fisik dan diagnosa pasti. Kode ICD-10 akan diberikan oleh dokter setelah pemeriksaan

Contoh Kinerja ke 2

kepala saya pusing dan badan saya panas, apakah saya harus meminum paracetamol?

            **🤖 Asisten Medis** (Waktu respon: 3.59s):
            ### Diagnosis Kemungkinan
Kepala pusing dan badan panas bisa seperti mobil yang overheat. Mesinnya (tubuh Anda) bekerja terlalu keras dan butuh didinginkan. Bisa karena demam, infeksi ringan, atau kelelahan. Tapi, bisa juga karena hal lain yang lebih serius, jadi kita perlu memastikannya.

💊 Rekomendasi Obat:

Paracetamol (atau Acetaminophen) bisa membantu menurunkan demam dan mengurangi rasa sakit kepala. Ikuti petunjuk penggunaan pada kemasan. Biasanya dosis untuk dewasa adalah 500mg – 1000mg setiap 4-6 jam, maksimal 4000mg dalam sehari. Jangan melebihi dosis yang dianjurkan.

🏥 Konsultasi Spesialis:

Jika gejala Anda memburuk (misalnya demam tinggi yang tidak turun, pusing yang hebat, muntah-muntah, atau muncul gejala lain seperti kaku kuduk), sebaiknya segera konsultasi ke dokter umum.

⚠️ Peringatan:

Hasil ini bukan diagnosis pasti. Segera hubungi dokter jika demam Anda lebih dari 3 hari, disertai sakit kepala hebat, kesulitan bernapas, rasa sakit dada, kejang, atau ruam kulit. Paracetamol juga dapat menyebabkan efek samping, baca informasi lengkap pada kemasan.

📌 Kode ICD-10: Tidak dapat diberikan tanpa pemeriksaan fisik. Kode ICD-10 bervariasi tergantung diagnosis pasti.

---

Penanganan Error dan Etika

Tiga kelas error utama:

SafetyFilterError: Prompt diblokir oleh sistem

ModelOverloadError: Keterbatasan memori GPU

MedicalChatError: Error umum lainnya dalam proses



---

Catatan Penggunaan

> Penting: model AI ini bukan pengganti diagnosis medis profesional. Selalu konsultasi dokter untuk penyesuaian kebutuhan anda.




---

Lisensi

Project ini open-source dan bebas digunakan untuk keperluan edukasi dan penelitian. Mohon sertakan atribusi apabila digunakan secara publik. License Apache 2.0


---

Kontributor

Riset pribadi
