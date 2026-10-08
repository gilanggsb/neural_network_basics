# 01 — Tabular Classification

> **Tujuan:** Membangun sistem klasifikasi dari data tabular (angka dan kategori) menggunakan arsitektur Neural Network standar dengan PyTorch.

## Konsep yang Dipelajari

- **Data Preprocessing** — Min-Max Scaling untuk menyamakan skala fitur numerik
- **Categorical Encoding** — Mengubah teks (misal "Bulanan"/"Tahunan") menjadi angka (Binary Encoding)
- **Multi-Class Classification** — Membuat AI yang bisa memilih lebih dari 2 kelas (CrossEntropyLoss)
- **Binary Classification** — Prediksi Ya/Tidak dengan Sigmoid + BCELoss
- **Model Caching** — Menyimpan dan memuat model yang sudah dilatih agar tidak training ulang

## Isi Folder

| Folder | Skenario | Tipe Masalah | Loss Function |
|---|---|---|---|
| `bank_loan/` | Persetujuan kredit bank berdasarkan gaji & utang | Binary (Diterima/Ditolak) | BCELoss |
| `rpg_ai/` | Klasifikasi aksi/karakter dalam game RPG | Multi-Class | CrossEntropyLoss |
| `smartphone_classification/` | Segmentasi HP ke kelas harga (Budget/Mid/Flagship) | Multi-Class (3 kelas) | CrossEntropyLoss |
| `factory_machine_monitoring/` | Pemantauan status mesin pabrik dari 3 sensor fisik | Multi-Class (4 kelas) | CrossEntropyLoss |
| `customer_churn/` | Prediksi pelanggan kabur berdasarkan data campuran | Binary (Kabur/Bertahan) | CrossEntropyLoss |

## Alur Standar Setiap Proyek

```
CSV File
  ↓
load_dataset()          → Baca & parse menjadi list Python
  ↓
Min-Max Scaling         → Normalisasi ke rentang 0.0 - 1.0
  ↓
torch.tensor()          → Ubah ke Tensor PyTorch
  ↓
nn.Module (Model)       → Definisi arsitektur hidden & output layer
  ↓
Training Loop           → optimizer, loss, backward, step
  ↓
save_model()            → Simpan bobot ke file .pth
  ↓
Inference               → Prediksi data baru
```

## Cheat Sheet: Kapan Pakai Apa?

| Kasus | Output Layer | Aktivasi Output | Loss |
|---|---|---|---|
| 2 kelas (Biner) | 1 neuron | Sigmoid | BCELoss |
| 2+ kelas | N neuron | Tidak ada (logits) | CrossEntropyLoss |
| Angka kontinu | 1 neuron | Tidak ada | MSELoss |

---
*Level 1 — Data Tabular: Numerik dan Kategorikal*
