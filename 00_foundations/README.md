# 00 — Neural Network Foundations

> **Tujuan:** Memahami cara kerja neural network dari level paling dasar sebelum menggunakan framework apapun.

## Konsep yang Dipelajari

- **Perceptron Manual** — Apa itu Weight, Bias, dan Dot Product tanpa library
- **Backpropagation** — Menghitung error dan memperbarui bobot secara manual dengan Python murni
- **Activation Function** — Kapan pakai Sigmoid vs ReLU dan mengapa
- **Logika Non-Linear (XOR)** — Mengapa satu neuron tidak cukup dan mengapa Hidden Layer itu penting
- **Training Loop PyTorch** — 4 pilar standar: Forward Pass → Loss → Backward → Optimizer Step

## Isi Folder

| File | Deskripsi |
|---|---|
| `neural_scratch.py` | Implementasi Perceptron & Backpropagation dari **nol tanpa library** (Pure Python) |
| `logic_gates_AND.py` | Single Neuron dengan PyTorch — Belajar logika AND (masalah linier) |
| `logic_gates_XOR.py` | Multi-Layer Perceptron — Memecahkan XOR yang tidak bisa diselesaikan satu neuron |

## Kunci Pemahaman

```
Input → [Weight · Bias] → Activation → Output
                 ↑
         Loss (Error)
                 ↑
        Backpropagation
                 ↑
        Optimizer.step()
```

## Prerequisite
- Python dasar (list, loop, fungsi)
- Matematika SMA: perkalian matriks sederhana dan konsep turunan/gradient

---
*Level 0 — Fondasi sebelum menggunakan PyTorch nn.Module*
