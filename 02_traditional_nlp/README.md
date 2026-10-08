# 02 — Traditional NLP (Bag-of-Words)

> **Tujuan:** Memahami cara paling dasar untuk membuat AI memproses teks — dan mengerti **mengapa pendekatan ini tidak cukup** untuk bahasa alami yang kompleks, sehingga kamu siap memahami *kenapa* Embedding dan Transformer lahir.

## Konsep yang Dipelajari

- **Bag-of-Words (BoW)** — Merepresentasikan teks sebagai vektor biner berdasarkan keberadaan kata kunci
- **Vocabulary Manual** — Membuat "kamus" kata sebagai fitur input AI
- **Kelemahan BoW** — Tidak memahami urutan kata, tidak memahami konteks ("tidak enak" ≠ "enak")
- **Klasifikasi Teks** — Memprediksi kategori dari teks yang belum pernah dilihat

## Isi Folder

| Folder | Skenario | Label | Vocab |
|---|---|---|---|
| `fake_news_detector/` | Deteksi berita hoaks vs asli berdasarkan kata kunci | Binary (Real/Fake) | 8 kata kunci |
| `sentimen_analysis/` | Analisis sentimen ulasan produk toko online | Multi-Class (Negatif/Netral/Positif) | 16 kata kunci |

## Cara Kerja Bag-of-Words

```
Kalimat: "pemerintah bagikan vaksin gratis"

Vocab = ["pemerintah", "vaksin", "gratis", "bahaya", "chip", "rahasia", ...]
                                                         
Vektor =  [    1.0    ,    1.0  ,    1.0  ,    0.0  ,   0.0 ,    0.0  , ...]
              ↑ Ada         ↑ Ada   ↑ Ada     ↑ Tidak ada
```

## Keterbatasan (Yang Akan Dipecahkan di Level Berikutnya)

| Masalah BoW | Solusi di Level Selanjutnya |
|---|---|
| Vocab harus dibuat manual | Sub-word Tokenization (BPE/WordPiece) otomatis |
| Tidak tahu urutan kata | Positional Encoding (Transformer) |
| Kata baru di luar vocab = 0 | Dense Word Embeddings |
| Tidak paham konteks | Self-Attention Mechanism |
| Kamus kecil = akurasi rendah | Pre-trained Large Language Models |

> 💡 **Proyek ini adalah jembatan.** Setelah memahami BoW dan keterbatasannya, kamu akan jauh lebih menghargai arsitektur Transformer di level berikutnya.

---
*Level 2 — NLP Tradisional: Fondasi sebelum Era Embedding dan Transformer*
