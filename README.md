# Neural Network & LLM Engineer Learning Path

Repositori ini adalah perjalanan belajar **dari nol hingga siap industri** mengikuti roadmap **Modern NLP / LLM Engineer**. Setiap folder merepresentasikan satu level kompetensi yang harus dikuasai secara berurutan.

---

## 🗺️ Roadmap

| Folder | Level | Topik | Status |
|---|---|---|---|
| [`00_foundations/`](./00_foundations/README.md) | 0 | Perceptron Manual & PyTorch Training Loop | ✅ Selesai |
| [`01_tabular_classification/`](./01_tabular_classification/README.md) | 1 | Neural Network untuk Data Tabular | ✅ Selesai |
| [`02_traditional_nlp/`](./02_traditional_nlp/README.md) | 2 | Bag-of-Words & Keterbatasannya | ✅ Selesai |
| [`03_tokenization_embeddings/`](./03_tokenization_embeddings/README.md) | 3 | BPE, WordPiece, Word Embeddings, RNN/LSTM | 🔲 Selanjutnya |
| [`04_transformers/`](./04_transformers/README.md) | 4 | Self-Attention, Positional Encoding, Mini-GPT | 🔲 Belum |
| [`05_rag_and_llm/`](./05_rag_and_llm/README.md) | 5 | HuggingFace, RAG, Vector DB, Evaluasi | 🔲 Belum |
| [`06_finetuning_and_agents/`](./06_finetuning_and_agents/README.md) | 6 | LoRA, QLoRA, Agents, LLMOps | 🔲 Belum |

---

## 📁 Struktur Proyek

```
neural_net_basics/
│
├── 00_foundations/                  ← Perceptron manual, XOR, AND
├── 01_tabular_classification/       ← Bank Loan, Churn, Pabrik, HP, RPG
├── 02_traditional_nlp/             ← Fake News, Sentimen (Bag-of-Words)
├── 03_tokenization_embeddings/     ← BPE, Embedding, RNN/LSTM
├── 04_transformers/                ← Self-Attention, Positional Enc, GPT mini
├── 05_rag_and_llm/                 ← HuggingFace, RAG, VectorDB
├── 06_finetuning_and_agents/       ← LoRA, QLoRA, Agents, LLMOps
│
├── helpers/                        ← Shared utilities (save/load model)
└── pyrightconfig.json
```

---

## 🛠️ Setup

```bash
# Buat virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy

# Jalankan contoh
python 00_foundations/logic_gates_AND.py
```

---

## 📚 Fondasi Konsep yang Perlu Dikuasai Sebelum Mulai

- **Weights & Bias** — Seberapa penting sebuah input dan kecenderungan neuron untuk aktif
- **Activation Function** — ReLU di hidden layer, Sigmoid di output biner, tanpa aktivasi untuk CrossEntropy
- **4 Pilar Training Loop:**
  1. `optimizer.zero_grad()` — Bersihkan gradient lama
  2. Forward Pass — Hitung output model
  3. `loss.backward()` — Hitung gradient (Backpropagation)
  4. `optimizer.step()` — Update bobot

---

## 🔗 Referensi

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)
- [Sebastian Raschka — Hands-On LLMs](https://github.com/rasbt/LLMs-from-scratch)
