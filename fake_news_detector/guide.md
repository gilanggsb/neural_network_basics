Boleh banget! Biar feel-nya lebih kerasa seperti proyek Machine Learning sungguhan, kita pakai file CSV yang isinya lebih banyak.

Kita masih di studi kasus yang sama: Detektor Judul Berita Hoaks.
Bedanya, sekarang kita akan buat program yang secara otomatis membaca file CSV dan secara otomatis mengecek kemunculan kata di Kamus (tidak diubah manual satu per satu).

---

Langkah 1: Siapkan File CSV
Buat file bernama dataset_berita.csv dan copy-paste data di bawah ini.
(Format: Judul Berita, Label | Label 0 = Asli, 1 = Hoaks)

Judul,Label
pemerintah bagikan vaksin gratis bulan ini,0
vaksin pemerintah dijamin aman untuk warga,0
berita baik hari ini vaksin gratis mulai beredar,0
warga senang dapat vaksin gratis,0
bahaya vaksin mengandung chip pelacak,1
chip pemerintah tersembunyi di vaksin,1
awas bahaya chip dalam vaksin gratis,1
terbongkar rahasia pemerintah pasang chip pelacak,1
pemerintah resmi luncurkan program vaksin,0
bahaya efek samping chip pemerintah,1


Langkah 2: Kamus Kata (Vocabulary)
Kita sepakati Kamus Kunci (Vocab) yang akan jadi patokan AI kita terdiri dari 8 kata ini.
Kamus: ["pemerintah", "vaksin", "gratis", "aman", "bahaya", "chip", "pelacak", "rahasia"]

---

🧩 Tantangan Kodingmu (Clue & Arahan)

Targetmu adalah merakit skrip PyTorch yang membaca file di atas, melakukan Training, lalu menebak 2 berita baru yang belum pernah ia lihat.

Tahap 1: Membaca & Mengubah Teks ke Angka (Bag-of-Words)
1. Buka file .csv dan lewati baris pertama (header).
2. Buat looping untuk membaca setiap baris kalimat.
3. Clue Logika Bag-of-Words: 
   Untuk setiap kalimat, buat array kosong [0, 0, 0, 0, 0, 0, 0, 0].
   Lalu cek: Apakah kata pertama di Kamus ("pemerintah") ada di kalimat itu? Kalau ada, ubah index 0 jadi 1.0. Apakah kata kedua ("vaksin") ada? Kalau ada, ubah index 1 jadi 1.0, dst.
   (Hint Python: Kamu bisa pakai operator if "vaksin" in kalimat:)

Tahap 2: Merakit Model PyTorch
1. Input: Berapa jumlah nilai input yang masuk ke Neural Network? (Lihat jumlah kata di Kamus).
2. Hidden Layer: Bebas, bisa pakai aturan Power of 2 (misal 16 neuron). Tambahkan ReLU.
3. Output & Activation: Hati-hati! Ini Deteksi Asli/Hoaks (Binary). Pastikan jumlah Output dan Fungsi Aktivasinya cocok dengan Cheat Sheet.

Tahap 3: Pelatihan (Training)
1. Gunakan Loss Function yang dikhususkan untuk keluaran Binary dengan Sigmoid.
2. Gunakan optim.Adam dengan lr=0.01 agar cepat konvergen. Epoch sekitar 1000 - 2000 biasanya cukup.

Tahap 4: Ujian Akhir (Inference)
Buat programmu menebak 2 judul ini:
- teks_uji_1 = "pemerintah berikan vaksin gratis yang aman"
- teks_uji_2 = "awas bahaya rahasia chip pelacak"

(Ingat: Kedua kalimat ujian ini harus kamu proses dulu pakai logika Bag-of-Words yang sama seperti Tahap 1 sebelum dilempar ke model(x)).

---

Silakan dicoba! Proses mengubah teks kalimat menjadi array berisikan 0 dan 1 (Tahap 1) mungkin akan mengasah logika looping Python kamu. 

Kasih tahu saya kalau kamu butuh hint di bagian pengolahan teksnya atau bagian desain arsitekturnya!