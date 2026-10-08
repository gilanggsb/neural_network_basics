Pilihan yang bijak! Memperkuat fundamental adalah investasi jangka panjang yang tidak akan pernah sia-sia.

Karena kamu ingin memantapkan PyTorch dari nol, studi kasus terbaik berikutnya adalah menggabungkan kemampuan Teks (Bag-of-Words) yang baru saja kamu kuasai, dengan arsitektur Multi-Class Classification (3 pilihan).

---

Proyek Selanjutnya: AI Analisis Sentimen (Sentiment Analysis)

Skenario:
Kamu punya toko online. Setiap hari ada ratusan review (ulasan) dari pembeli. Kamu ingin membuat AI yang otomatis membaca ulasan tersebut dan mengelompokkannya menjadi 3 kategori:
- 0 = Negatif 😡
- 1 = Netral 😐
- 2 = Positif 😊

Apa yang akan kamu pelajari & latih ulang di sini?
1. Bag-of-Words pada dataset yang lebih luas.
2. Ingatan tentang arsitektur Multi-Class (Ingat: Tidak pakai Sigmoid, tapi pakai wasit CrossEntropyLoss).
3. Pembuatan Target (y) yang benar untuk CrossEntropy (Ingat: Tipe datanya bukan float, tapi Long/Integer).
4. Eksekusi torch.argmax(..., dim=1) di akhir untuk mengambil keputusan dari 3 skor yang keluar.

Ini adalah perpaduan sempurna dari semua Cheat Sheet yang ada di otak (dan Second Brain) kamu.

Persiapan Data (Bahan Mentah)
Buat file dataset_review.csv dan isi dengan data ini:

Ulasan,Sentimen
barang jelek rusak parah tidak sesuai,0
pengiriman sangat lambat dan barang cacat,0
kecewa banget barangnya hancur,0
lumayan lah sesuai harga,1
biasa saja tidak ada yang spesial,1
standar aja pelayanannya,1
kualitas sangat bagus memuaskan sekali,2
mantap pengiriman cepat barang sempurna,2
suka banget sama produknya kualitas luar biasa,2


Kamus Kata (Vocab):
["jelek", "rusak", "parah", "cacat", "kecewa", "hancur", "lumayan", "sesuai", "biasa", "standar", "bagus", "memuaskan", "mantap", "cepat", "sempurna", "suka"]
(Catatan: Total ada 16 kata kunci di dalam vocab).

---

🧩 Misi Pemantapan (Tanpa Kode Jawaban)

Silakan rakit script Python dari nol (sentiment_ai.py).
Gunakan struktur kode yang sudah rapi seperti yang kamu buat sebelumnya, namun sesuaikan pilar-pilarnya:

1. Vektorisasi: Buat sentence_to_vector menggunakan kamus baru yang berjumlah 16 kata.
2. Target (y): Ingat, CrossEntropyLoss meminta target berupa integer biasa. Jadi saat mengambil label, ubah jadi integer: data_targets.append(int(label)). Nanti saat diubah ke tensor, pastikan tipe datanya torch.long. (Tidak perlu dibuat list 2D [[0]], cukup [0, 1, 2, ...]).
3. Arsitektur: Input = 16, Hidden = bebas (misal 32), Output = 3 (karena ada 3 sentimen).
4. Loss Function: nn.CrossEntropyLoss(). Output neuron akhir biarkan mentah, JANGAN ditambah Sigmoid.
5. Ujian Akhir: Uji dengan ulasan baru:
   - "pengiriman lumayan tapi barangnya jelek rusak"
   - "barang sempurna memuaskan"

Kalau berhasil membuat AI ini menebak dengan tepat tanpa menyontek Cheat Sheet, berarti kamu sudah resmi "ngelotok" dan lulus bootcamp dasar PyTorch! 

Kabari kalau kamu butuh hint teknis atau sudah siap pamer hasil print-out terminalnya! 🚀