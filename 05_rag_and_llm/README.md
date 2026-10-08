# 05 — RAG & LLM Applications

> **Tujuan:** Menyambungkan LLM raksasa dengan data privat dan dokumen lokal — kompetensi inti seorang LLM Engineer di industri.

## Status: 🔒 Belum Dimulai

**Prerequisite:** Selesaikan folder `04_transformers` dan punya pemahaman dasar tentang vector database.

---

## Roadmap Materi (Level 3 & 4 — Modern NLP/LLM Engineer)

### Level 3 — Ekosistem Hugging Face & Open-Weights

#### Menjalankan SLM Lokal
- Mencari dan mendownload Small Language Model (SLM) dari Hugging Face Hub
- Menjalankan model ringan (Llama/Mistral versi quantized) secara lokal
- Memahami format model: GGUF, safetensors

#### Pretrained Embedding Models
- Embedding specialist models: BGE, E5, `text-embedding-3-small` (OpenAI)
- Kapan embedding model berbeda dari model generatif?

---

### Level 4 — RAG (Retrieval-Augmented Generation)

#### Chunking Strategy
Memotong dokumen panjang menjadi potongan kecil tanpa merusak makna.

```
Dokumen PDF (50 halaman)
         ↓
    Chunking
   (overlap)
         ↓
[chunk_1][chunk_2][chunk_3]...
         ↓
   Embedding tiap chunk
         ↓
    Vector Database
```

- **Fixed-size chunking** — Sederhana, bisa kehilangan konteks
- **Semantic chunking** — Potongan berdasarkan makna
- **Overlap** — Menghindari kehilangan informasi di batas chunk

#### Vector Database & Hybrid Search
- **Semantic Search** — Mencari berdasarkan kesamaan makna (cosine similarity)
- **BM25** — Pencarian kata kunci tradisional (TF-IDF)
- **Hybrid Search** — Menggabungkan keduanya untuk hasil terbaik

#### Reranking
Menyortir ulang hasil pencarian dengan model yang lebih presisi sebelum dikirim ke LLM.

#### Structured Output
Memaksa LLM menjawab dalam format JSON yang terstruktur menggunakan function calling atau grammar constraints.

#### Evaluasi LLM
- **LLM-as-a-judge** — Menggunakan LLM lain untuk menilai kualitas jawaban
- **RAGAS** — Framework evaluasi RAG: faithfulness, answer relevancy, context precision

---

## Proyek yang Akan Dibuat

| Proyek | Konsep | Status |
|---|---|---|
| `huggingface_basics/` | Load & inference SLM dari Hugging Face | 🔲 Belum |
| `chunking_strategy/` | Implementasi berbagai strategi chunking | 🔲 Belum |
| `simple_rag/` | RAG pipeline sederhana end-to-end | 🔲 Belum |
| `hybrid_search/` | Kombinasi Vector + BM25 | 🔲 Belum |
| `structured_output/` | JSON output dari LLM | 🔲 Belum |
| `rag_evaluation/` | Evaluasi dengan RAGAS | 🔲 Belum |

---
*Level 5 — Aplikasi LLM & RAG: Data Privat Bertemu Kecerdasan Buatan*
