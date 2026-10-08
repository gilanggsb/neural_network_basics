# 03 — Tokenization & Embeddings

> **Tujuan:** Memahami bagaimana model modern mengubah teks menjadi representasi matematika yang kaya makna — berbeda jauh dari Bag-of-Words yang kaku.

## Status: 🔒 Belum Dimulai

**Prerequisite:** Selesaikan folder `00_foundations`, `01_tabular_classification`, dan `02_traditional_nlp` terlebih dahulu.

---

## Roadmap Materi (Level 1 — Modern NLP/LLM Engineer)

### 1. Sub-word Tokenization
Pengganti Bag-of-Words yang lebih cerdas. AI modern tidak memisahkan teks per kata, tapi per **sub-kata** (morfem).

- **BPE (Byte Pair Encoding)** — Dipakai GPT-2, GPT-3, Llama
- **WordPiece** — Dipakai BERT, Google
- Keunggulan: Menangani kata baru (Out-of-Vocabulary) secara elegan

**Contoh BPE:**
```
"pembelajaran" → ["pembel", "##ajar", "##an"]
"unbelievable" → ["un", "##believe", "##able"]
```

### 2. Word Embeddings (nn.Embedding)
Mengubah token ID menjadi **vektor N-dimensi** yang membawa makna semantik.

- "Raja" - "Pria" + "Wanita" ≈ "Ratu" (operasi vektor bermakna!)
- Setiap token direpresentasikan sebagai koordinat dalam ruang N-dimensi
- PyTorch: `nn.Embedding(vocab_size, embedding_dim)`

### 3. RNN & LSTM (Sejarah & Keterbatasannya)
Arsitektur sebelum Transformer — membaca teks **secara berurutan** dengan memori.

- **RNN** — Memori sederhana, tapi "lupa" konteks awal pada kalimat panjang
- **LSTM** — Perbaikan RNN dengan Gate (Forget/Input/Output), tapi tetap punya masalah *bottleneck*
- Mengapa ditinggalkan: Tidak bisa diparalelkan dan bottleneck pada teks panjang

---

## Proyek yang Akan Dibuat

| Proyek | Konsep | Status |
|---|---|---|
| `bpe_tokenizer/` | Implement BPE dari scratch | 🔲 Belum |
| `word_embedding/` | nn.Embedding + visualisasi vektor | 🔲 Belum |
| `rnn_sequence/` | Klasifikasi teks dengan RNN/LSTM | 🔲 Belum |

---
*Level 3 — Representasi Teks Modern: Dari Token ke Vektor Bermakna*
