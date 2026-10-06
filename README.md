---
title: Fundamental Neural Network dan PyTorch
date: 2026-10-05
tags: [machine-learning, pytorch, neural-network, second-brain, python]
source: Dev Lab Session
---

# Fundamental Neural Network & PyTorch

Dokumen ini adalah rangkuman perjalanan membangun Artificial Intelligence dari titik nol (Perceptron manual) hingga menggunakan *framework* PyTorch untuk sistem klasifikasi multi-layer.

## 1. Komponen Inti Neural Network
- **Weights (Bobot):** Seberapa penting sebuah nilai input memengaruhi hasil akhir.
- **Bias:** Kecenderungan (sifat bawaan) neuron untuk aktif, mencegah neuron kaku saat input bernilai 0.
- **Dot Product (Summation):** Perhitungan `(Input * Bobot) + Bias`. Di PyTorch, operasi ini dibungkus secara rahasia dan efisien di dalam `nn.Linear`.

## 2. Activation Function (Kapan Pakai Apa?)
Fungsi aktivasi membengkokkan hasil perhitungan matematika linier agar AI bisa memahami pola yang rumit (non-linear).
- **ReLU (Rectified Linear Unit):** 
  - *Aturan:* Jika hasil negatif, jadikan 0. Jika positif, biarkan.
  - *Lokasi:* Digunakan di **Hidden Layer** (Layer Tengah).
  - *Fungsi:* Sangat cepat dan mencegah *Vanishing Gradient* pada model yang dalam.
  - *Bahaya:* Jika dipakai di Single Layer / Output Layer yang nilainya kebetulan negatif semua di awal, bisa terjadi **Dying ReLU** (Neuron mati/Loss *stuck* karena gradient menjadi 0 mutlak).
- **Sigmoid:**
  - *Aturan:* Memampatkan angka menjadi rentang `0.0` sampai `1.0` membentuk kurva S.
  - *Lokasi:* Digunakan HANYA di **Output Layer** untuk kasus *Binary Classification* (Klasifikasi 2 Pilihan, misal: Ya/Tidak, Setuju/Tolak).
  - *Fungsi:* Mengubah hasil mentah menjadi nilai probabilitas (persentase keyakinan).

## 3. Siklus Standar PyTorch (4 Pilar)
Semua model PyTorch, dari yang sederhana sampai ChatGPT, mengikuti pola aliran ini:
1. **Data Flow (Vektor/Matriks):** Data dilempar bersamaan (Batch Processing) menggunakan Tensor, bukan dilooping satu-satu.
2. **Forward Pass:** Proses perhitungan dari Input -> Hidden Layer -> Output Layer.
3. **Loss (Hitung Error):** Mengevaluasi tebakan AI dengan kunci jawaban menggunakan `criterion` (misal: `nn.MSELoss()`).
4. **Training Loop & Gradient Descent:**
   - `optimizer.zero_grad()`: Bersihkan catatan gradient lama.
   - `loss.backward()`: Hitung turunan/arah koreksi dari error (*Backpropagation* otomatis).
   - `optimizer.step()`: Eksekusi perubahan Bobot dan Bias secara perlahan berdasarkan *Learning Rate*.

## 4. Evaluasi & Memori AI (Inference)
- **`torch.no_grad()`:** Mematikan mesin pencatat gradient PyTorch. Digunakan saat menguji AI (Inference) agar kalkulasi jauh lebih cepat dan hemat RAM karena AI tidak sedang belajar.
- **`.item()`:** Mengekstrak angka murni Python dari dalam objek Tensor.
- **Dimana AI menyimpan ilmunya?** AI tidak menghafal data (tidak butuh database SQL untuk data latih). AI menyimpan kepintarannya pada nilai akhir **Weights dan Bias** di dalam memori RAM komputer (variabel model).
- **Menyimpan Model:** Agar AI tidak amnesia saat program ditutup, kepintaran (Bobot & Bias) tersebut disimpan ke dalam file menggunakan perintah:
  `torch.save(model.state_dict(), "model_cerdas.pth")`

---
*Catatan Perjalanan:*
- Berhasil membuat Neuron Manual dari nol (Python murni).
- PyTorch Linear Regression untuk logika AND.
- PyTorch Multi-Layer Perceptron (MLP) memecahkan tantangan sejarah **XOR**.
- PyTorch sistem prediksi persetujuan kredit Bank (Credit Scoring) dengan pola relasional Gaji & Utang.
