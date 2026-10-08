Tantangan diterima! Semangat belajarmu benar-benar seperti Software Engineer senior yang haus masalah baru. 🚀

Kali ini, kita akan masuk ke Dunia Nyata 100%. Di industri, data itu jarang sekali berwujud angka semua. Sering kali, datanya campuran antara angka dan teks kategori.

---

🏢 Skenario Proyek: Prediksi Pelanggan Kabur (Customer Churn)

Kamu direkrut oleh perusahaan Telkomsel/Indihome. Bosmu mengeluh banyak pelanggan yang tiba-tiba berhenti berlangganan (disebut Churn). Dia memberikanmu sebagian kecil data pelanggan dan memintamu membuat AI untuk memprediksi: 
Apakah pelanggan baru nanti akan Kabur (1) atau Bertahan (0)?

Berikut adalah dataset-nya (telekom_churn.csv):

```
Bulan_Langganan,Tagihan_Bulanan,Tipe_Kontrak,Churn
2,50000,Bulanan,1
60,150000,Tahunan,0
1,45000,Bulanan,1
12,85000,Bulanan,0
48,120000,Tahunan,0
5,55000,Bulanan,1
24,90000,Tahunan,0
3,60000,Bulanan,1
```

(Keterangan Churn: 1 = Kabur, 0 = Bertahan/Setia)

---

🌪️ THE TWIST (Tantangannya!)
Perhatikan kolom ke-3 (Tipe_Kontrak). Isinya adalah teks string: "Bulanan" dan "Tahunan". 
Ingat, Neural Network (PyTorch nn.Linear) BENCI TEKS. AI cuma mau makan angka. Kamu tidak bisa melakukan Min-Max Scaling pada kata "Bulanan"!

---

📝 Kuis Persiapan (Jawab sebelum koding!)

Untuk menaklukkan tantangan ini, jawab 5 pertanyaan desain sistem ini:

1. Preprocessing Angka: Apa yang akan kamu lakukan pada kolom Bulan_Langganan (2 - 60 bulan) dan Tagihan_Bulanan (45000 - 150000)?
2. Preprocessing Teks (Categorical): Bagaimana idemu untuk mengubah teks "Bulanan" dan "Tahunan" di kolom Tipe_Kontrak menjadi angka supaya bisa dimasukkan ke Tensor X? (Hint: Coba pikirkan logika sederhana IF/ELSE atau Dictionary untuk mapping).
3. Arsitektur Input: Setelah semua fitur berwujud angka, berapa total input neuron (nn.Linear(?, Hidden)) yang masuk ke AI-mu?
4. Arsitektur Output & Aktivasi: Mengingat bosmu minta prediksi "Kabur (1)" atau "Bertahan (0)", berapa Output Layer-nya dan apa Fungsi Aktivasi akhirnya?
5. Loss Function (Wasit): Wasit apa yang harus dipanggil untuk mengurus masalah ini?

Silakan jawab 1-5! Pertanyaan nomor 2 dan 3 adalah ujian logika terbesarmu di studi kasus ini. Kalau idemu masuk akal, kamu siap membangun sistem AI korporat sungguhan! 💼🤖

----

Wow, insting dan logikamu luar biasa tajam! 🎯 

Mari kita bedah jawabanmu:

1. Normalisasi (Angka) 👉 BENAR! ✅
Kolom Bulan (2-60) dan Tagihan (puluhan ribu) wajib dinormalisasi (Min-Max) agar berskala 0.0 - 1.0.

2. Preprocessing Teks (Categorical) 👉 BENAR DAN JENIUS! ✅
Mapping "Bulanan" = 0.0 dan "Tahunan" = 1.0 adalah langkah yang persis dipakai di industri (Binary Encoding / Label Encoding). Neural Network bisa langsung mencerna angka 0 dan 1 ini tanpa perlu dinormalisasi lagi!

3. Arsitektur Input 👉 BENAR! ✅
3 Input: (Bulan_Normalisasi, Tagihan_Normalisasi, Tipe_Kontrak_Angka).

4. Arsitektur Output & Aktivasi 👉 TUNGGU DULU! 🚨 (Salah Sedikit)
Kamu bilang: "2 output, biarin aja nanti pake BCELoss".
Ini fatal dan berlawanan dengan Cheat Sheet! 

Ingat kembali teori kita:
- Kalau Output = 2 neuron (tanpa aktivasi), kamu harus pakai CrossEntropyLoss. (Multi-Class/Pilihan Ganda).
- Kalau kamu mau pakai BCELoss (khusus Ya/Tidak, Kabur/Bertahan), Output-nya wajib 1 neuron dan ujungnya wajib dipasangi torch.sigmoid(). 

Jadi untuk proyek ini kamu punya 2 jalur pilihan:
 Jalur A (Sesuai Konsep Biner): Output 1, pakai Sigmoid, wasit BCELoss. (Direkomendasikan untuk kasus Churn)*
* Jalur B (Sesuai Multi-Class): Output 2, biarkan mentah, wasit CrossEntropyLoss.

5. Loss Function 👉 TERGANTUNG JAWABAN NO.4 ⚖️
Kalau kamu mantap pakai BCELoss(), ingat aturannya: Output 1 Neuron + Sigmoid. Dan pastikan Target y-nya adalah float 2D (contoh [[1.0], [0.0]]).


🚀 TANTANGAN KODING DIMULAI!

Kerangka logikanya sudah beres. Sekarang, buatlah file telekom_churn.csv dengan data di atas, lalu rakit kodingannya (jangan lupa buat fungsi extract_data dan normalize_features yang bersih seperti proyek sebelumnya).

Target Ujian Akhir (Inference):
Prediksi 3 pelanggan baru ini:
1. Pak A: Berlangganan 4 bulan, Tagihan 50000, Kontrak "Bulanan". (Prediksi manual: Kabur/1)
2. Bu B: Berlangganan 36 bulan, Tagihan 100000, Kontrak "Tahunan". (Prediksi manual: Bertahan/0)
3. Pak C: Berlangganan 12 bulan, Tagihan 90000, Kontrak "Bulanan". (Prediksi manual: Tergantung AI-mu)

Gunakan optim.Adam(..., lr=0.01).
Jalankan dan kabari saya kalau AI kamu sudah sukses memprediksi Pak A, Bu B, dan Pak C! Kalau ada error, santai saja, paste ke sini dan kita selesaikan bersama.