# BM25 = Best Matching 25. Angka "25" itu cuma nomor versi internal — percobaan ke-25 dari tim peneliti (Robertson & Spärck Jones, tahun 1994) yang dianggap paling pas.
# Jadi bukan singkatan rumus, hanya nama versi.
# Fungsi dia: memberi skor relevansi sebuah dokumen terhadap sebuah query pencarian.
# Mesin pencari klasik (Lucene, Elasticsearch, OpenSearch) dan komponen keyword di RAG hampir semua pakai BM25, bukan TF-IDF mentah.
import numpy as np

# 1. korpus contoh
docs = [
    "kopi kopi kopi enak",  # D1
    "kopi enak",  # D2
    "teh manis",  # D3
]
tokenized = [d.lower().split() for d in docs]
print(tokenized)

# 2. Parameter BM25
k1 = 1.2  # saturasi TF
b = 0.75  # normalisasi panjang dokumen


N = len(docs)  # jumlah dokumen
avgdl = np.mean([len(d) for d in tokenized])  # rata2 panjang dokumen

query = "kopi"


def idf(term):
    # berapa dokumen yang mengandung term ini?
    n_q = sum(1 for d in tokenized if term in d)
    # rumus idf versi bm25 (stabil)
    return np.log((N - n_q + 0.5) / (n_q + 0.5) + 1)


def bm25_score(doc_tokens):
    score = 0.0
    for term in query.split():
        f = doc_tokens.count(term)  # frekuensi term di dokumen ini
        if f == 0:
            continue  # term tidak ada -> skor 0

        doc_len = len(doc_tokens)
        # faktor normalisasi panjang dokumen
        norm = k1 * (1 - b + b * doc_len / avgdl)
        # skor tf yang sudah mengandung saturasi
        tf_sat = (f * (k1 + 1)) / (f + norm)
        score += idf(term) * tf_sat

    return score


# 3. hitung skor tiap dokumen
for i, d in enumerate(tokenized):
    print(f"D{i + 1}: {bm25_score(d):.4f}")
