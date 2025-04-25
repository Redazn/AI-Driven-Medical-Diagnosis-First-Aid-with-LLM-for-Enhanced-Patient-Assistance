
import numpy as np
from evaluate import load
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import google.generativeai as genai
import torch
from IPython.display import display, Markdown
import ipywidgets as widgets

# ===== 1. Inisialisasi Model =====
def load_models():
    try:
        # BioBERT
        biobert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

        # BioGPT
        biogpt = AutoModelForCausalLM.from_pretrained(
            "microsoft/biogpt",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")

        return biobert, biobert_tokenizer, biogpt, biogpt_tokenizer

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise

# ===== 2. Enhanced Medical Database =====
MEDICAL_DB = {
    "diabetes": {
        "gejala": ["haus berlebihan", "luka sulit sembuh", "kaki nyeri", "penglihatan kabur"],
        "obat": ["metformin", "insulin", "glibenclamide"],
        "spesialis": "Endokrinologi",
        "kode_icd": "E11"
    },
    "hipertensi": {
        "gejala": ["sakit kepala", "pusing", "mimisan", "nyeri dada"],
        "obat": ["amlodipine", "captopril", "losartan"],
        "spesialis": "Kardiologi",
        "kode_icd": "I10"
    }
}

# ===== 3. Advanced Error Handling =====
class MedicalChatError(Exception):
    pass

class SafetyFilterError(MedicalChatError):
    pass

class ModelOverloadError(MedicalChatError):
    pass

def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except genai.types.BlockedPromptError:
            raise SafetyFilterError("Konten diblokir: Pertanyaan mengandung konten sensitif")
        except torch.cuda.OutOfMemoryError:
            raise ModelOverloadError("GPU overload: Silakan gunakan pertanyaan lebih pendek")
        except Exception as e:
            raise MedicalChatError(f"Error sistem: {str(e)}")
    return wrapper

# ===== 4. Simplified Benchmarking Tools =====
def simple_evaluate(prediction, reference):
    """Evaluasi sederhana tanpa dependency tambahan"""
    pred_lower = prediction.lower()
    ref_lower = reference.lower()

    # Hitung akurasi kata kunci
    keywords = ref_lower.split()
    matches = sum(1 for kw in keywords if kw in pred_lower)
    keyword_accuracy = matches / len(keywords) if keywords else 0

    return {
        "keyword_accuracy": keyword_accuracy,
        "exact_match": int(pred_lower == ref_lower)
    }

# ===== 5. Enhanced Medical Function =====
@handle_errors
def answer_medical_question(question):
    # Langkah 1: Ekstraksi Entitas Medis
    inputs = biobert_tokenizer(question, return_tensors="pt").to(biobert.device)
    with torch.no_grad():
        outputs = biobert(**inputs)

    # Langkah 2: Deteksi Kondisi
    tokens = biobert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    symptoms = [token for token in tokens if token not in ["[CLS]", "[SEP]", "?", "."]]

    possible_conditions = []
    for condition, data in MEDICAL_DB.items():
        symptom_matches = [s for s in symptoms if s in data["gejala"]]
        if len(symptom_matches) >= 2:
            possible_conditions.append((
                condition,
                len(symptom_matches)/len(data["gejala"]),
                symptom_matches
            ))

    # Langkah 3: Generasi Jawaban
    if possible_conditions:
        condition_info = "\n".join([
            f"{cond[0]} (Kecocokan: {cond[1]*100:.1f}%)\n"
            f"- Gejala Terdeteksi: {', '.join(cond[2])}\n"
            f"- Kemungkinan Obat: {', '.join(MEDICAL_DB[cond[0]]['obat'])}"
            for cond in sorted(possible_conditions, key=lambda x: x[1], reverse=True)[:2]
        ])

        prompt = f"""
        [INSTRUKSI MEDIS]
        1. Identifikasi kondisi dari gejala: {', '.join(symptoms)}
        2. Kondisi yang mungkin: {condition_info}
        3. Berikan penjelasan untuk pasien dengan:
           - Bahasa sederhana
           - Format ICD-10
           - Peringatan konsultasi dokter
        """
    else:
        prompt = f"Jawab pertanyaan medis: {question}\nGunakan bahasa awam."

    # Generasi dengan BioGPT
    inputs = biogpt_tokenizer(prompt, return_tensors="pt").to(biogpt.device)
    output = biogpt.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    draft = biogpt_tokenizer.decode(output[0], skip_special_tokens=True)

    # Langkah 4: Penyempurnaan dengan Gemini
    try:
        genai.configure(api_key="API_KEY_ANDA")
        model = genai.GenerativeModel('gemini-1.5-flash')

        response = model.generate_content(f"""
        Anda adalah dokter senior. Sederhanakan ini untuk pasien:

        [DRAFT ASLI]
        {draft}

        [FORMAT OUTPUT]
        ### Diagnosis Kemungkinan
        (Jelaskan dengan analogi sehari-hari)

        💊 Rekomendasi Obat:
        (Hanya obat OTC, sertakan dosis umum)

        🏥 Konsultasi Spesialis:
        (Spesialis yang relevan)

        ⚠️ Peringatan:
        "Hasil ini bukan diagnosis pasti. Segera hubungi dokter jika..."

        📌 Kode ICD-10: (jika tersedia)
        """)

        return response.text

    except Exception as e:
        raise MedicalChatError(f"Error pada Gemini: {str(e)}")

# ===== 6. Benchmarking System =====
TEST_CASES = [
    {
        "input": "Saya sering haus dan penglihatan kabur",
        "expected_output": "diabetes",
        "expected_keywords": ["haus", "penglihatan kabur"]
    },
    {
        "input": "Kepala saya sakit dan mimisan",
        "expected_output": "hipertensi",
        "expected_keywords": ["sakit kepala", "mimisan"]
    }
]

def run_benchmark():
    results = []
    for case in TEST_CASES:
        try:
            response = answer_medical_question(case["input"])
            evaluation = simple_evaluate(
                response.lower(),
                case["expected_output"]
            )

            # Deteksi keywords
            detected_keywords = [
                kw for kw in case["expected_keywords"]
                if kw in response.lower()
            ]

            results.append({
                "test_case": case["input"],
                "keyword_accuracy": evaluation["keyword_accuracy"],
                "exact_match": evaluation["exact_match"],
                "keywords_detected": len(detected_keywords),
                "response": response[:100] + "..."  # Potong response untuk display
            })

        except Exception as e:
            results.append({
                "test_case": case["input"],
                "error": str(e)
            })

    return pd.DataFrame(results)

# ===== 7. Enhanced UI =====
def create_ui():
    text_input = widgets.Textarea(placeholder="Deskripsikan gejala Anda...")
    output = widgets.Output()

    def on_submit(btn):
        with output:
            output.clear_output()
            display(Markdown(f"**👨⚕️ Pasien**: {text_input.value}"))

            try:
                start_time = time.time()
                response = answer_medical_question(text_input.value)
                latency = time.time() - start_time

                display(Markdown(f"""
                **🤖 Asisten Medis** (Waktu respon: {latency:.2f}s):
                {response}
                """))

            except MedicalChatError as e:
                display(Markdown(f"**❌ Error**: {str(e)}"))
            except Exception as e:
                display(Markdown("**⚠️ Sistem sibuk**. Silakan coba lagi nanti."))

    submit = widgets.Button(description="Diagnosis", button_style='success')
    submit.on_click(on_submit)

    display(widgets.VBox([
        widgets.HTML("<h2 style='color: #3b82f6'>🩺 MedicalAI Chatbot</h2>"),
        widgets.HTML("<i>Untuk pertanyaan medis non-darurat</i>"),
        text_input,
        submit,
        output
    ]))

# ===== 8. Main Execution =====
if __name__ == "__main__":
    print("🔍 Memuat model...")
    biobert, biobert_tokenizer, biogpt, biogpt_tokenizer = load_models()

    print("🧪 Menjalankan benchmark...")
    benchmark_results = run_benchmark()
    display(benchmark_results)

    print("🚀 Launching UI...")
    create_ui()
