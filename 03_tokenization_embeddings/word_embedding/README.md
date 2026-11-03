# Word embedding dan cosine similarity

Dua demo ini mengikuti satu kata dari vocab sampai vektor embedding, lalu
menunjukkan bagaimana cosine similarity berubah saat vektor `apel` digeser
ke arah makna makanan.

## Menjalankan

Pastikan `torch` dan `numpy` tersedia di Python aktif. Setup virtualenv ada di
[README utama](../../README.md). Jalankan dari folder ini:

```bash
python 01_embedding_lookup.py
python 02_cosine_similarity.py
```

## Demo 1: token ID dan tabel embedding

Vocab dimulai dengan `<pad>: 0`; kata `ia` mendapat ID `5`. Tabel embedding
berisi 7 baris dan 4 angka per baris, jadi shape-nya `(7, 4)`. Dua cara lookup
mengambil baris yang sama:

- `embedding.weight[token_id_ia]` menghasilkan shape `(4,)`.
- `embedding(input_ids)` untuk satu ID menghasilkan shape `(1, 4)`.
- `squeeze(0)` melepas dimensi batch berukuran satu agar hasilnya bisa
  dibandingkan dengan `torch.equal`.

### Seed dan nilai awal

`torch.manual_seed(42)` mengatur pengacak PyTorch sebelum tabel dibuat. Dengan
seed dan setup yang sama, nilai awal embedding bisa diulang. Angka `42` tidak
punya arti khusus dan tidak menentukan token ID. Mengubah seed mengubah nilai
awal, bukan bentuk tabel atau cara lookup.

Nilai vektor awal dibuat acak; demo ini belum melatihnya agar punya makna
semantik. Pembulatan di output hanya untuk memudahkan membaca angka.

### Shape

- `(7, 4)` berarti 7 baris token termasuk `<pad>`, dengan 4 fitur per vektor.
- `(4,)` berarti satu sumbu berisi empat angka.
- `(1, 4)` berarti satu item dalam batch, dengan empat angka.
- Jika dua ID masuk dengan `embedding_dim=4`, hasilnya berbentuk `(2, 4)`.

Istilah “embedding 4 dimensi” merujuk pada empat fitur. Shape `(4,)` tetap
punya satu sumbu tensor.

### Gradien

`grad_fn` yang terlihat pada tensor menunjukkan operasi yang dicatat autograd;
itu bukan nilai gradien. Gradien dihitung saat training memanggil `backward()`
dan disimpan di `.grad`. Demo ini tidak melakukan training.

### Padding

`<pad>` adalah token pengisi agar panjang urutan sama saat kalimat berbeda
panjang diproses bersama. Contohnya, `[5, 6]` dan `[5]` dapat ditulis sebagai
`[5, 6]` dan `[5, 0]`. Padding menyamakan panjang urutan, bukan dimensi
embedding. Dalam demo ini `<pad>` hanya ada di vocab dan tidak masuk ke input.

Nama `<pad>` saja belum memberi perlakuan khusus. `nn.Embedding` perlu
`padding_idx=0` untuk menandai baris itu sebagai padding. PyTorch menginisialisasi
baris tersebut ke nol dan tidak memperbaruinya lewat gradien.
[Dokumentasi `nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

## Demo 2: cosine similarity

Rumus cosine:

```text
cos(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Cosine membandingkan arah vektor. Mengalikan vektor dengan angka positif
mengubah panjang dan dot product dengan faktor yang sama, jadi cosine tetap.

Pada demo, `apel_awal = [0.5, 0.5]`. Dot product-nya dengan `makan = [0.9,
0.1]` adalah `0.45 + 0.05 = 0.5`. Dengan `perusahaan = [0.1, 0.9]`, hasilnya
`0.05 + 0.45 = 0.5`. Panjang dua kandidat juga sama karena koordinatnya
tertukar. Maka cosine awal seri, sekitar `0.781`.

Vektor baru dibentuk dengan bobot yang dipilih manual untuk demo:

```text
0.8 * [0.9, 0.1] + 0.2 * [0.5, 0.5]
= [0.72, 0.08] + [0.10, 0.10]
= [0.82, 0.18]
```

Cosine `apel_baru` ke `makan` sekitar `0.994`, sedangkan ke `perusahaan`
sekitar `0.321`. Angka `0.8` dan `0.2` bukan hasil cosine atau training;
keduanya hanya memperagakan pergeseran vektor.

## Tanya jawab belajar

| Pertanyaan | Jawaban |
|---|---|
| Apa yang berulang jika seed tetap? | Nilai awal embedding; ID token berasal dari vocab. |
| Kenapa dua lookup `ia` menghasilkan nilai sama? | Keduanya mengambil baris dengan ID `5`. |
| Apa yang dilakukan `squeeze(0)` pada `(1, 4)`? | Menghapus sumbu batch; empat nilai vektornya tetap. |
| Apakah `grad_fn` adalah gradien? | Bukan; itu jejak operasi untuk menghitung gradien nanti. |
| Apa nama parameter untuk menandai baris padding? | `padding_idx`; `num_embeddings` adalah jumlah baris tabel. |
| Kenapa cosine awal `apel` seri? | Dot product dan panjang kandidat sama. |
| Apa yang terjadi jika `makan` dikali 10? | Cosine tetap, karena arah vektornya tidak berubah. |
| Apakah bobot `0.8` dan `0.2` dipelajari? | Tidak, dipilih manual untuk demo. |

Demo self-attention ada di [Level 4](../../04_transformers/self_attention/README.md).
