import numpy as np

# 1. Dataset Dokumen (Corpus)
docs = ["saya suka kopi hitam", "saya suka teh manis", "saya beli kopi hitam"]

# 2. Tokenisasi Sederhana & Buat Vocabulary Unik
# Split per kata, jadikan huruf kecil semua
tokenized_docs = [doc.lower().split() for doc in docs]

# Ambil semua kata unik(sorted agar urutan kolomnya konsisten)
vocab = sorted({word for doc in tokenized_docs for word in doc})

print("Vocabulary (Urutan Kolom)")
print(vocab)

# 3. Hitung TF (Term Frequency)
# Rumus: (Jumlah kemunculan kata di dokumen) / (Total kata di dokumen tersebut)
tf_matrix = []
for doc in tokenized_docs:
    doc_len = len(doc)
    tf_row = [doc.count(word) / doc_len for word in vocab]
    tf_matrix.append(tf_row)

tf_matrix = np.array(tf_matrix)

# 4. Hitung IDF(Inverse Document Frequency)
# Rumus: log(Total dokumen / jumlah dokumen yang mengandung kata tersebut)
N = len(docs)
df = np.zeros(len(vocab))

for i, word in enumerate(vocab):
    # hitung berapa banyak dokumen mengandung kata ini
    df[i] = sum(1 for doc in tokenized_docs if word in doc)

# Gunakan log natural (np.log)
# Perhatikan: jika df == N (muncul di semua doc), N/df = 1, maka log(1) = 0!
idf = np.log(N / df)

# 5. Hitung Skor Akhir: TF-IDF = TF * IDF# Perkalian element-wise (broadcasting NumPy otomatis menyesuaikan shape)
tfidf = tf_matrix * idf

# 6. Tampilkan Hasilprint("\n--- Nilai IDF (Tingkat Kelangkaan Kata) ---")
for word, score in zip(vocab, idf):
    print(f"{word:7s}: {score:.4f}")

print("\n--- Matriks TF-IDF (Baris = Dokumen, Kolom = Kata) ---")
print(np.round(tfidf, 4))
