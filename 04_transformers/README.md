# 04 — Transformers & Self-Attention

> **Tujuan:** Memahami arsitektur revolusioner yang menjadi fondasi semua LLM modern — dari BERT hingga GPT-4.

## Status: 🔒 Belum Dimulai

**Prerequisite:** Selesaikan folder `03_tokenization_embeddings` terlebih dahulu.

---

## Roadmap Materi (Level 2 — Modern NLP/LLM Engineer)

### 1. Self-Attention Mechanism
Inti dari Transformer. AI bisa **melihat seluruh kalimat sekaligus** dan memutuskan bagian mana yang paling relevan.

```
"Bank di tepi sungai itu sangat indah"
       ↑
"Bank" memperhatikan "sungai" lebih kuat dari "uang" → konteks yang benar!
```

- **Query, Key, Value** — Tiga komponen kunci mekanisme Attention
- **Scaled Dot-Product Attention** — `softmax(QK^T / √dk) · V`
- **Multi-Head Attention** — Beberapa "kepala" perhatian yang belajar aspek berbeda

### 2. Positional Encoding
Transformer tidak membaca berurutan → harus diberi info posisi kata secara eksplisit.

- Sinusoidal Encoding (orisinal dari paper "Attention is All You Need")
- Rotary Position Embedding (RoPE) — dipakai Llama

### 3. Arsitektur Decoder-Only
Dasar dari keluarga model GPT (Generative Pre-trained Transformer).

- **Causal Masking** — Token hanya bisa melihat token sebelumnya (tidak boleh "curang" melihat ke depan)
- **Language Modeling** — Prediksi token berikutnya
- Berbeda dengan Encoder (BERT) yang bisa melihat dua arah

### 4. Mini-GPT dari Scratch
Mengimplementasikan kerangka nanoGPT (ala Andrej Karpathy) untuk benar-benar memahami setiap komponen.

---

## Proyek yang Akan Dibuat

| Proyek | Konsep | Status |
|---|---|---|
| `self_attention/` | Implementasi Self-Attention dari nol | 🔲 Belum |
| `positional_encoding/` | Visualisasi Positional Encoding | 🔲 Belum |
| `mini_gpt/` | Decoder-only mini Transformer (nanoGPT style) | 🔲 Belum |

---
*Level 4 — Revolusi Self-Attention: Arsitektur yang Melahirkan Era LLM*
