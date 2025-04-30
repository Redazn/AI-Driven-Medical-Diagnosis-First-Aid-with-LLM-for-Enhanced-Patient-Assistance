
# -*- coding: utf-8 -*-
"""
MEDICAL CHATBOT WITH RAG AND CONFIDENCE SCORING - OPTIMIZED VERSION
"""

# ===== 1. INSTALASI PAKET =====
# !pip install transformers torch google-generativeai ipywidgets sacremoses rouge-score faiss-cpu sentence-transformers datasets evaluate plotly

# ===== 2. IMPORT LIBRARY =====
import numpy as np
import evaluate
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import google.generativeai as genai
import torch
from IPython.display import display, Markdown
import ipywidgets as widgets
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
import os
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# ===== 3. ERROR HANDLING =====
class MedicalChatError(Exception):
    """Base class for medical chatbot errors"""
    pass

class SafetyFilterError(MedicalChatError):
    """Error for blocked content"""
    pass

class ModelOverloadError(MedicalChatError):
    """Error for GPU overload"""
    pass

def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "blocked" in str(e).lower() or "safety" in str(e).lower():
                raise SafetyFilterError("Konten diblokir: Pertanyaan mengandung konten sensitif")
            elif isinstance(e, torch.cuda.OutOfMemoryError):
                raise ModelOverloadError("GPU overload: Silakan gunakan pertanyaan lebih pendek")
            elif "invalid literal for int()" in str(e):
                raise MedicalChatError("Format data tidak valid - silakan coba pertanyaan lain")
            else:
                raise MedicalChatError(f"Error sistem: {str(e)}")
    return wrapper

# ===== 4. MODEL LOADING =====
def load_models():
    """Memuat semua model yang diperlukan"""
    try:
        print("🔍 Memuat BioBERT untuk entity recognition...")
        biobert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

        print("🧠 Memuat BioGPT untuk generasi teks...")
        biogpt = AutoModelForCausalLM.from_pretrained(
            "microsoft/biogpt",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")

        print("📚 Memuat Sentence Transformer untuk embeddings...")
        embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        return biobert, biobert_tokenizer, biogpt, biogpt_tokenizer, embedder

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise

# ===== 5. RAG DATASET SOLUTION =====
def setup_medical_rag(embedder):
    """Menyiapkan database medis dengan fallback options"""
    try:
        # Coba beberapa dataset medis gratis secara berurutan
        dataset_options = [
            ("medquad", None),  # Pertanyaan-jawaban medis
            ("medalpaca/medical_meadow_medical_flashcards", None),  # Flashcards medis
            ("medalpaca/medical_meadow_wikidoc", None)  # Artikel medis
        ]

        for ds_name, config in dataset_options:
            try:
                print(f"🔎 Mencoba dataset: {ds_name}...")
                dataset = load_dataset(ds_name, config, split='train')
                break
            except:
                continue
        else:
            raise Exception("Semua dataset gagal diakses")

        # Proses dataset yang berhasil di-load
        df = dataset.to_pandas()

        # Handle berbagai format dataset
        if 'question' in df.columns and 'answer' in df.columns:
            df = df[['question', 'answer']]
        elif 'input' in df.columns and 'output' in df.columns:  # Format Alpaca
            df = df.rename(columns={'input': 'question', 'output': 'answer'})
        elif 'context' in df.columns and 'answer' in df.columns:  # Format SQuAD
            df = df.rename(columns={'context': 'question'})

        # Filter data kosong
        df = df[(df['question'].str.len() > 0) & (df['answer'].str.len() > 0)]

        # Buat embeddings
        print("🔧 Membuat embeddings...")
        embeddings = embedder.encode(df['question'].tolist(), show_progress_bar=True).astype('float32')

        # Buat FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Mapping ID ke teks
        id_to_text = {int(i): (str(q), str(a)) for i, (q, a) in enumerate(zip(df['question'], df['answer']))}

        print(f"✅ Database medis siap! ({len(df)} entri)")
        return index, id_to_text

    except Exception as e:
        print(f"❌ Error setting up RAG: {str(e)}")
        print("🔄 Menggunakan dataset contoh minimal...")
        # Fallback dataset minimal
        samples = [
            ("Apa gejala diabetes?", "Gejala diabetes: sering haus, sering buang air kecil, lemas."),
            ("Bagaimana mengatasi demam?", "Istirahat, minum air, kompres hangat, parasetamol."),
            ("Pertolongan pertama sesak nafas?", "Tenangkan pasien, cari udara segar, hubungi dokter.")
        ]
        questions = [q for q, a in samples]
        embeddings = embedder.encode(questions).astype('float32')
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        id_to_text = {int(i): samples[i] for i in range(len(samples))}
        return index, id_to_text

# ===== 6. OPTIMIZED CONFIDENCE SCORING SYSTEM =====
def calculate_confidence(response, context, question, embedder, cache={}):
    """Enhanced confidence calculation with caching and better metrics"""
    cache_key = f"{hash(response)}-{hash(context)}"
    if cache_key in cache:
        return cache[cache_key]

    # 1. Semantic similarity with context
    resp_embed = embedder.encode(response)
    ctx_embed = embedder.encode(context)
    semantic_sim = np.dot(resp_embed, ctx_embed) / (np.linalg.norm(resp_embed) * np.linalg.norm(ctx_embed))

    # 2. Response completeness
    question_keywords = set([w.lower() for w in question.split() if len(w) > 3])
    response_words = set([w.lower() for w in response.split()])
    completeness = len(question_keywords.intersection(response_words)) / max(1, len(question_keywords))

    # 3. Medical terminology presence
    medical_terms = {
        "high": ["diagnosis", "treatment", "symptoms", "clinical", "prognosis", "pathology"],
        "medium": ["drug", "dose", "medication", "disease", "condition", "therapy"],
        "low": ["health", "medical", "doctor", "hospital", "patient", "care"]
    }

    weights = {"high": 1.0, "medium": 0.7, "low": 0.4}
    term_scores = []

    for priority, terms in medical_terms.items():
        matches = sum(term.lower() in response.lower() for term in terms)
        if len(terms) > 0:
            term_scores.append(weights[priority] * (matches / len(terms)))

    medical_term_score = sum(term_scores) / len(term_scores) if term_scores else 0

    # 4. Uncertainty analysis
    uncertainty_phrases = [
        ("might be", 0.7), ("may be", 0.7), ("possibly", 0.6),
        ("potentially", 0.6), ("could be", 0.6), ("appears to be", 0.5),
        ("suggest", 0.4), ("consider", 0.4), ("uncertain", 0.9),
        ("not clear", 0.8), ("unclear", 0.8), ("unknown", 0.7)
    ]

    uncertainty_score = 0
    matches = 0
    for phrase, weight in uncertainty_phrases:
        if phrase in response.lower():
            uncertainty_score += weight
            matches += 1

    uncertainty = uncertainty_score / max(1, matches) if matches > 0 else 0

    # 5. Length appropriateness
    length_ratio = len(response) / (len(context) + 1e-6)
    length_score = 1.0 if 0.2 <= length_ratio <= 1.5 else max(0, 1 - abs(length_ratio - 0.85) / 2)

    # 6. Structure assessment
    structure_indicators = [
        "\n-", "\n•", "\n1.", "\n2.", "Pertama", "Kedua", "Terakhir",
        "Gejala:", "Penyebab:", "Pengobatan:"
    ]
    structure_score = min(1.0, sum(indicator in response for indicator in structure_indicators) / 3)

    # Combined confidence
    confidence = (
        0.35 * semantic_sim +
        0.20 * completeness +
        0.25 * medical_term_score +
        0.15 * (1 - uncertainty) +
        0.05 * length_score +
        0.10 * structure_score
    )

    result = {
        "confidence_score": max(0, min(1, confidence)),
        "metrics": {
            "semantic_similarity": round(semantic_sim, 2),
            "completeness": round(completeness, 2),
            "medical_terms": round(medical_term_score, 2),
            "certainty": round(1 - uncertainty, 2),
            "length_appropriateness": round(length_score, 2),
            "structure": round(structure_score, 2)
        }
    }

    cache[cache_key] = result
    return result

# ===== 7. MODIFIED ANSWER FUNCTION =====
@handle_errors
def answer_medical_question(question, biobert, biobert_tokenizer, biogpt, biogpt_tokenizer, embedder, rag_index, id_to_text):
    try:
        if not isinstance(question, str):
            question = str(question)

        # Langkah 1: Retrieval dengan RAG
        query_embedding = embedder.encode(question)
        _, indices = rag_index.search(np.array([query_embedding]).astype('float32'), k=3)

        indices = np.array(indices).flatten()

        contexts = []
        for idx in indices:
            if idx >= 0 and idx in id_to_text:
                q, a = id_to_text[idx]
                contexts.append(f"Q: {q}\nA: {a}")

        if not contexts:
            raise MedicalChatError("Tidak ditemukan konteks medis yang relevan")

        medical_context = "\n\n".join(contexts)

        # Langkah 2: Generasi jawaban
        prompt = f"""Berdasarkan konteks:\n{medical_context}\n\nJawab pertanyaan:\n{question}\n\nDengan:
        - Bahasa sederhana
        - Berikan rekomendasi pertolongan pertama
        - Sertakan ketidakpastian jika perlu
        - Sarankan konsultasi dokter"""

        inputs = biogpt_tokenizer(prompt, return_tensors="pt").to(biogpt.device)
        output = biogpt.generate(
            **inputs,
            max_new_tokens=300,
            num_beams=3,
            top_p=0.85,
            do_sample=True,
            temperature=0.7,
            output_scores=True,
            return_dict_in_generate=True
        )

        probs = torch.softmax(output.scores[0], dim=-1)
        avg_prob = probs.mean().item()

        draft = biogpt_tokenizer.decode(output.sequences[0], skip_special_tokens=True)

        # Langkah 3: Hitung confidence baru
        confidence = calculate_confidence(draft, medical_context, question, embedder)

        # Langkah 4: Refinement dengan Gemini
        try:
            genai.configure(api_key="AIzaSyAzejYOIcl_0xkPRjZVNaWUkUB0-ZL29Pk")  # Ganti dengan API key
            model = genai.GenerativeModel('gemini-1.5-flash')
            refined = model.generate_content(f"Sebagai dokter senior yang paham etika, sederhanakan ini:\n{draft}")
            final_response = refined.text

            confidence = calculate_confidence(final_response, medical_context, question, embedder)

            return {
                "response": final_response,
                "confidence": confidence["confidence_score"],
                "confidence_metrics": confidence["metrics"],
                "generation_prob": avg_prob
            }
        except:
            return {
                "response": draft,
                "confidence": confidence["confidence_score"],
                "confidence_metrics": confidence["metrics"],
                "generation_prob": avg_prob
            }
    except Exception as e:
        raise MedicalChatError(f"Error processing question: {str(e)}")

# ===== 8. MODIFIED BENCHMARKING =====
TEST_CASES = [
    {
        "input": "Apa gejala umum diabetes?",
        "expected_keywords": ["gula darah", "haus", "poliuria"]
    },
    {
        "input": "Bagaimana cara mengatasi demam?",
        "expected_keywords": ["parasetamol", "istirahat", "cairan"]
    }
]

def run_benchmark(biobert, biobert_tokenizer, biogpt, biogpt_tokenizer, embedder, rag_index, id_to_text):
    results = []
    for case in TEST_CASES:
        try:
            start_time = time.time()
            result = answer_medical_question(
                case["input"],
                biobert, biobert_tokenizer,
                biogpt, biogpt_tokenizer,
                embedder, rag_index, id_to_text
            )
            latency = time.time() - start_time

            detected_keywords = [
                kw for kw in case["expected_keywords"]
                if kw.lower() in result["response"].lower()
            ]
            accuracy = len(detected_keywords)/len(case["expected_keywords"])

            results.append({
                "test_case": case["input"],
                "accuracy": f"{accuracy:.0%}",
                "confidence": f"{result['confidence']:.0%}",
                "keywords_found": ", ".join(detected_keywords),
                "response": result["response"][:100] + "...",
                "latency": f"{latency:.2f}s"
            })
        except Exception as e:
            results.append({
                "test_case": case["input"],
                "error": str(e)
            })

    return pd.DataFrame(results)

# ===== 9. UPDATED UI WITH NEW METRICS =====
def create_ui(biobert, biobert_tokenizer, biogpt, biogpt_tokenizer, embedder, rag_index, id_to_text):
    text_input = widgets.Textarea(placeholder="Masukkan pertanyaan medis...")
    output = widgets.Output()
    confidence_gauge = widgets.FloatProgress(
        value=0,
        min=0,
        max=1,
        description='Confidence:',
        bar_style='info',
        style={'bar_color': '#FFA500'},
        orientation='horizontal'
    )

    def update_confidence_display(confidence, metrics):
        if confidence > 0.7:
            confidence_gauge.bar_style = 'success'
            confidence_gauge.style.bar_color = 'green'
        elif confidence > 0.4:
            confidence_gauge.bar_style = 'warning'
            confidence_gauge.style.bar_color = 'orange'
        else:
            confidence_gauge.bar_style = 'danger'
            confidence_gauge.style.bar_color = 'red'
        confidence_gauge.value = confidence

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            name='Confidence Metrics'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Confidence Metrics Breakdown"
        )
        display(fig)

    def on_submit(btn):
        with output:
            output.clear_output()
            question = text_input.value.strip()
            if not question:
                return

            display(Markdown(f"**🧑⚕️ Pertanyaan**: {question}"))

            try:
                start_time = time.time()
                result = answer_medical_question(
                    question,
                    biobert, biobert_tokenizer,
                    biogpt, biogpt_tokenizer,
                    embedder, rag_index, id_to_text
                )
                latency = time.time() - start_time

                update_confidence_display(result["confidence"], result["confidence_metrics"])

                if result["confidence"] < 0.4:
                    warning = "⚠️ **Warning**: Jawaban mungkin kurang akurat (confidence rendah)"
                elif result["confidence"] < 0.7:
                    warning = "⚠️ **Note**: Jawaban mungkin memerlukan verifikasi"
                else:
                    warning = "✅ **High Confidence**"

                display(Markdown(f"""
                **🤖 Jawaban** ({latency:.2f}s | Confidence: {result['confidence']:.0%})
                {warning}
                {result['response']}
                """))

                with widgets.Accordion(selected_index=None):
                    display(widgets.VBox([
                        widgets.HTML("<b>Detail Confidence Metrics:</b>"),
                        widgets.HTML(f"Semantic Similarity: {result['confidence_metrics']['semantic_similarity']:.2f}"),
                        widgets.HTML(f"Completeness: {result['confidence_metrics']['completeness']:.2f}"),
                        widgets.HTML(f"Medical Terms: {result['confidence_metrics']['medical_terms']:.2f}"),
                        widgets.HTML(f"Certainty: {result['confidence_metrics']['certainty']:.2f}"),
                        widgets.HTML(f"Structure: {result['confidence_metrics']['structure']:.2f}")
                    ]))

            except Exception as e:
                display(Markdown(f"**❌ Error**: {str(e)}"))

    submit = widgets.Button(description="Dapatkan Jawaban", button_style='success')
    submit.on_click(on_submit)

    display(widgets.VBox([
        widgets.HTML("<h1 style='color: #3b82f6'>🩺 MedicalAI (Confidence Monitoring)</h1>"),
        widgets.HTML("<i>Sistem informasi medis dengan confidence scoring</i>"),
        text_input,
        submit,
        confidence_gauge,
        output
    ]))

# ===== 10. MAIN EXECUTION =====
if __name__ == "__main__":
    print("🚀 Inisialisasi sistem dengan confidence monitoring...")

    print("🔧 Memuat model AI...")
    biobert, biobert_tokenizer, biogpt, biogpt_tokenizer, embedder = load_models()

    print("📊 Membangun database medis...")
    rag_index, id_to_text = setup_medical_rag(embedder)

    print("🧪 Menjalankan benchmark...")
    benchmark_results = run_benchmark(
        biobert, biobert_tokenizer,
        biogpt, biogpt_tokenizer,
        embedder, rag_index, id_to_text
    )
    display(benchmark_results)

    print("🖥️ Menjalankan UI...")
    create_ui(
        biobert, biobert_tokenizer,
        biogpt, biogpt_tokenizer,
        embedder, rag_index, id_to_text
    )