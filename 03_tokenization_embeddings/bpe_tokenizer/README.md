# Skeleton BPE karakter-level

Folder ini disiapkan sebagai latihan. Algoritmanya sengaja belum ditulis penuh:
isi TODO sendiri, satu langkah per satu waktu. Kalau mentok, kirim bagian yang
kamu tulis; aku akan memberi hint bertahap sebelum menunjukkan solusi.

Semua contoh hanya butuh Python standard library. Jalankan perintah dari root
repo setelah TODO di file tersebut selesai:

```bash
python 03_tokenization_embeddings/bpe_tokenizer/step01_bpe_training_demo.py
python 03_tokenization_embeddings/bpe_tokenizer/step02_bpe_tokenizer.py
python 03_tokenization_embeddings/bpe_tokenizer/step03_bpe_analysis.py
```

## Urutan latihan dan checkpoint

1. **Training manual** — isi mapping karakter dan IDs awal, hitung stats yang
   mencakup pasangan overlapping, lalu lakukan tiga merge kiri-ke-kanan.
   Output tiap ronde harus menunjukkan stats, pemenang, token baru, dan IDs baru.
2. **Tokenizer** — isi metode train, encode, dan decode. Pastikan kata baru yang
   hurufnya dikenal bisa dipecah, lalu cek round-trip untuk semua kalimat korpus
   dan beberapa kalimat uji.
3. **Analisis** — tampilkan ukuran vocab dan merge awal/akhir. Bandingkan kata
   sering dengan kata langka atau typo; kata sering seharusnya cenderung lebih
   ringkas.

## Konsep -> Lihat di file mana

| Konsep | File |
|---|---|
| Stats pasangan termasuk overlapping | step01_bpe_training_demo.py |
| Merge kiri-ke-kanan | step01_bpe_training_demo.py |
| Training, vocab, dan urutan merge | step02_bpe_tokenizer.py |
| Encode, decode, dan anti-UNK untuk huruf yang dikenal | step02_bpe_tokenizer.py |
| Kata populer dibanding kata langka/typo | step03_bpe_analysis.py |

## Penyederhanaan dibanding minbpe/GPT

- Vocab awal dari karakter unik di korpus, bukan 256 byte.
- Tanpa regex pre-tokenization, jadi spasi juga dapat ikut tergabung ke token.
- Tanpa special tokens.
- Contoh berfokus pada alfabet yang muncul di korpus Bahasa Indonesia.
  Kata baru bisa dipecah jika semua karakternya dikenal; karakter baru di luar
  alfabet awal perlu ditangani tersendiri.
- Aturan untuk memutuskan frekuensi seri boleh dibuat deterministik agar hasil
  demo mudah diulang.

Referensi yang dibaca: [karpathy/minbpe](https://github.com/karpathy/minbpe),
khususnya [contoh tiga merge di README](https://github.com/karpathy/minbpe#quick-start),
[minbpe/basic.py](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py),
dan [lecture.md](https://github.com/karpathy/minbpe/blob/master/lecture.md).

## Pertanyaan refleksi

1. Kenapa stats dapat menghitung dua pasangan aa pada aaa, tetapi merge kiri-ke-kanan hanya mengganti satu?
2. Bagaimana ukuran target vocab memengaruhi panjang tokenisasi?
3. Apa batas anti-UNK pada tokenizer karakter-level yang vocab awalnya hanya mengambil karakter dari korpus?

## Catatan quiz dan jawaban Gilang

### Quiz: memilih merge dengan ID terkecil

**Pertanyaan:** Jika `available = [(3, (0, 1)), (4, (1, 2))]`, kandidat mana yang dipilih `min(available)`?

**Jawabanmu:** `(0, 1)`, karena `3 < 4`.

**Catatan:** Benar. Angka pertama adalah ID merge; ID lebih kecil berarti merge itu dipelajari lebih dulu.

### Quiz: OOV dan urutan merge

1. **Pertanyaan:** Kenapa `misuh` bisa di-encode walau kata itu tidak ada di korpus?

   **Jawabanmu:** “Karena per character kaya m i s u h sudah ada tokennya makanya bisa di encode.”

   **Catatan:** Benar. Lebih tepatnya, setiap hurufnya sudah ada di `char_to_id`, jadi tokenizer bisa memulai dari ID tiap karakter.

2. **Pertanyaan:** Kenapa `encode()` memilih kandidat dengan ID merge terkecil?

   **Jawabanmu:** “Nah ini lupa kwkw.”

   **Catatan setelah dibahas:** ID merge diberikan berurutan saat training. ID lebih kecil berarti merge dipelajari lebih dulu, dan `encode()` mengikuti urutan itu.

3. **Pertanyaan:** Apa yang terjadi kalau kata baru mengandung huruf yang tidak ada di `self.char_to_id`?

   **Jawabanmu:** “Error pasti.”

   **Catatan:** Benar. Lookup karakter yang tidak dikenal menghasilkan `KeyError` pada implementasi ini.

### Quiz: Analisis dan Tradeoff BPE (step03)

1. **Pertanyaan:** Kenapa BPE selalu mempelajari pola potongan kecil (`an`, `er`) terlebih dahulu sebelum potongan kata yang lebih panjang (`belajar`)?

   **Jawabanmu:** “sebenernya dia tergantung seberapa sering kata tersebut muncul pas di training, jadi kalo makin sering kata tersebut muncul ada kemungkinan bakal jadi 1”

   **Catatan:** Tepat. Selain frekuensi, ada sifat bottom-up: BPE bekerja secara iteratif menggabungkan pasangan karakter. Kata panjang seperti `belajar` belum bisa digabung sebelum komponen-komponennya (misal `bela` dan `jar`) terbentuk lewat merge-merge sebelumnya.

2. **Pertanyaan:** Kenapa kata typo `"bagnet"` dipecah menjadi karakter individual, padahal mungkin ada pasangan huruf yang mirip di korpus?

   **Jawabanmu:** “karena vocab sizenya cuma 86 pastinya ga semua pair masuk”

   **Catatan:** Benar. Dengan batas vocab 86, hanya pasangan yang paling sering muncul di korpus yang sempat dimerge. Pasangan spesifik seperti `('b', 'a')` atau `('e', 't')` tanpa spasi mungkin tidak cukup sering muncul di korpus training untuk masuk kuota 86 token.

3. **Pertanyaan:** Kenapa LLM modern (GPT-4, Llama 3) memilih vocab sekitar 32k–128k, bukan 500 atau 2.000.000?

   **Jawabanmu:** “itu tergantung seberapa kuat spek buat training mereka, karena kalo vocab size kecil dia beban di komputasi sedangkan terlalu besar bebannya di RAM”

   **Catatan:** Sangat tepat. 32k–128k adalah sweet spot trade-off: sekuens token tidak terlalu panjang (menghindari bottleneck komputasi self-attention $O(N^2)$), tetapi matriks embedding dan output head ($V \times d$) juga tidak membengkak berlebihan di memori GPU/VRAM.


