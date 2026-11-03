# Self-attention manual: dua ronde

Demo ini melanjutkan pembahasan embedding ke Q, K, V dan self-attention
sederhana. Perhitungannya memakai satu head dan tidak memakai scaling.

## Menjalankan

Pastikan `numpy` tersedia di Python aktif. Setup virtualenv ada di
[README utama](../../README.md). Dari root repo jalankan:

```bash
python 04_transformers/self_attention/03_hand_rolled_attention.py
```

Untuk lookup dan cosine sebelum attention, lihat
[demo embedding](../../03_tokenization_embeddings/word_embedding/README.md).

## Vektor dan matriks manual

Dimensi pertama menggambarkan sifat agresor/pengejar; dimensi kedua
menggambarkan sifat korban/takut. Contoh vektor yang dipakai:

| Kata | Vektor |
|---|---|
| kucing | `[2, 0]` |
| tikus | `[-2, 2]` |
| ia | `[0, 0]` |
| takut | `[0, 10]` |
| lapar | `[0, -10]` |

`W_query` dan `W_value` adalah matriks identitas. `W_key` dipilih manual
sebagai `[[0, -0.5], [0, 0]]`. Dengan vektor baris dan perkalian `X @ W_key`,
kucing menjadi Key `[0, -1]` dan tikus menjadi Key `[0, 1]`.

Untuk setiap kata, kode menghitung:

```text
Q = X @ W_query
K = X @ W_key
V = X @ W_value
```

`@` adalah operator perkalian matriks dan setara dengan `np.matmul`. Operator
`*` mengalikan elemen yang posisinya sama.

## Ronde pertama: kumpulkan konteks

Vektor awal `ia` adalah `[0, 0]`. Karena `W_query` identitas, `Q_ia` juga
`[0, 0]`. Dot product Q ini terhadap semua Key menghasilkan enam skor nol.
Softmax mengubahnya menjadi enam bobot yang sama, masing-masing `1/6`.

Vektor `baru_ia` adalah jumlah `bobot * V` untuk semua kata. Karena bobotnya
seragam, operasi ini sama dengan mean pooling V:

- `ia takut` → `baru_ia = [0.167, 2]`
- `ia lapar` → `baru_ia = [0.167, -1.333]`

Bobotnya sama, tetapi V untuk kata terakhir berbeda (`takut = [0, 10]` dan
`lapar = [0, -10]`). Karena itu hasil agregasi dan konteks `ia` berbeda.

## Ronde kedua: Query mengikuti konteks

Query baru dihitung dari vektor konteks:

```text
Q_ia_baru = baru_ia @ W_query
```

Karena `W_query` identitas, Query baru sama dengan `baru_ia`. Skor Q terhadap
semua Key lalu dilewatkan ke softmax. Hasil terbesarnya:

| Kalimat | Kata dengan bobot terbesar | Bobot |
|---|---|---:|
| `ia takut` | TIKUS | `0.678` |
| `ia lapar` | KUCING | `0.421` |

Alur utamanya: V berbeda → konteks `ia` berbeda → Query ronde 2 berbeda →
skor dan bobot perhatian berubah.

## Apa yang disederhanakan di demo

`W_query`, `W_key`, dan `W_value` dipilih manual agar alur mudah dilihat.
Transformer sungguhan mempelajari matriks proyeksi ini saat training. Untuk
setiap masukan, model menghitung skor Q–K dan softmax menghasilkan bobot
perhatian yang dinamis. Bobot perhatian bukan satu nilai tetap yang disimpan
untuk semua kalimat.

Softmax di script menstabilkan perhitungan dengan mengurangi skor maksimum
sebelum eksponensial:

```text
exp(skor - max(skor)) / sum(exp(skor - max(skor)))
```

## Tanya jawab belajar

| Pertanyaan | Jawaban |
|---|---|
| Apa arti `@`? | Perkalian matriks; setara dengan `np.matmul`. |
| `bobot_r2` dihitung dari mana? | `softmax(skor_r2)`. |
| Jika `W_key` berubah, apa yang terdampak? | K berubah, lalu skor dan bobotnya dapat berubah. |
| Kenapa Q ronde 2 berbeda untuk takut dan lapar? | Nilai V terakhir berbeda, sehingga konteks `ia` berbeda. |
| Apakah matriks proyeksi demo ini dipelajari? | Tidak, semuanya dipilih manual untuk latihan. |
