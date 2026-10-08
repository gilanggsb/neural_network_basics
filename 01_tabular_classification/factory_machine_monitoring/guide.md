Tantangan diterima! Kali ini saya tidak akan memberikan kode full-nya, melainkan memberikan Skenario Proyek, data mentah, dan 5 Pertanyaan Kuis yang harus kamu jawab sebelum kamu boleh mulai coding.

Ini akan menguji insting AI Engineer kamu! 🕵️‍♂️

---

Skenario Usecase: AI Pemantau Mesin Pabrik 🏭

Kamu disewa oleh sebuah pabrik untuk membuat AI yang memantau kesehatan mesin raksasa pembuat mobil. Setiap mesin dipasangi 3 sensor pendeteksi.
1. Suhu Mesin (dalam Celcius, rentang 50 - 150°C)
2. Getaran (dalam Hertz, rentang 10 - 100 Hz)
3. Suara (dalam Desibel, rentang 40 - 120 dB)

Kamu harus membuat AI yang mengkategorikan kondisi mesin menjadi 4 Status:
- 0 = Normal 🟢
- 1 = Waspada 🟡 (Butuh perawatan minggu depan)
- 2 = Bahaya 🟠 (Harus dimatikan hari ini)
- 3 = Rusak Parah 🔴 (Sudah meledak/hancur)

Berikut adalah contoh isi dataset (mesin_pabrik.csv):

```
suhu,getaran,suara,status
60,15,50,0
65,20,55,0
85,45,75,1
90,50,80,1
115,75,95,2
120,80,100,2
145,95,115,3
150,100,120,3
```


---

📝 Kuis Arsitektur AI (Jawab ini dulu!)

Berdasarkan skenario dan bentuk data di atas, tolong jawab 5 pertanyaan ini (boleh dijawab langsung di chat ini):

1. Logika Data (Preprocessing): Datamu berupa angka puluhan dan ratusan (bukan teks/kata). Langkah wajib apa yang harus kamu lakukan pada ketiga data sensor tersebut sebelum dimasukkan ke Tensor X?
2. Arsitektur Linear Layer: Di nn.Linear pertama (Input) dan terakhir (Output), berapa angka yang harus kamu masukkan? (Misal: Input = ?, Output = ?).
3. Fungsi Aktivasi Output: Di fungsi forward, apa yang harus kamu lakukan pada output layer? (Pakai Sigmoid, ReLU, atau Biarkan mentah?)
4. Criterion (Loss Function): Apa nama wasit (Loss) yang paling tepat untuk masalah ini?
5. Optimizer: Mengingat ini adalah data tabular (angka baris-kolom) sederhana dan kita ingin AI cepat belajar, siapa "sopir" yang akan kamu tugaskan?

Silakan jawab 1 sampai 5. Kalau tebakan arsitekturmu benar, berarti kamu sudah siap tempur merakit kodenya! 🔥


Mari kita bahas dan luruskan jawaban nomor 4 dan 5:

---

Koreksi & Pembahasan Kuis:

1. Normalisasi (Min-Max Scaling) 👉 BENAR! ✅
Karena angka sensor (150°C, 100 Hz, 120 dB) sangat beragam, mereka wajib dipampatkan ke skala 0.0 - 1.0 supaya tidak ada sensor yang mendominasi saat perhitungan bobot.

2. Arsitektur Linear Layer 👉 BENAR! ✅
- Input: 3 (karena ada 3 sensor: Suhu, Getaran, Suara).
- Output: 4 (karena ada 4 status mesin: Normal, Waspada, Bahaya, Rusak).
- Hidden Layer: 8 adalah pilihan yang sangat solid berdasarkan Rule of Thumb (lebih besar dari input, tapi tidak berlebihan untuk data kecil).

3. Fungsi Aktivasi Output 👉 BENAR! ✅
- Hidden Layer wajib pakai ReLU agar bisa memahami batas-batas sensor yang rumit (non-linear).
- Output Layer dibiarkan Mentah (Logits) karena ini adalah kasus klasifikasi banyak pilihan.

4. Criterion (Loss Function) 👉 Ini Jawaban Benarnya! 💡
Karena mesin pabrik ini diklasifikasikan ke dalam 4 Pilihan (Multi-Class Classification), maka kamu WAJIB menggunakan:
👉 nn.CrossEntropyLoss()

Pengingat Cheat Sheet:
- Kalau hanya tebak angka harga 👉 MSELoss()
- Kalau 2 pilihan (Misal cuma Normal / Rusak) 👉 BCELoss() + Sigmoid.
- Kalau Lebih dari 2 pilihan (Normal/Waspada/Bahaya/Rusak) 👉 CrossEntropyLoss() + Tanpa Aktivasi.

5. Optimizer 👉 Pilih Salah Satu (Adam)! 🚗
Kamu menulis "Adam SGD". Sebenarnya ini adalah 2 "sopir" yang berbeda:
- optim.SGD (Sopir lama, manual, lambat tapi teliti).
- optim.Adam (Sopir baru, otomatis, ngebut).
Untuk proyek ini, pilihlah optim.Adam(model.parameters(), lr=0.01) karena lebih cepat menemukan pola data angka.

---

Misi Eksekusi (Tantangan Koding)

Sekarang blueprint (kerangka kerja)-nya sudah jelas! 
Silakan buat file mesin_pabrik.csv dengan data contoh di atas, lalu rakit kodenya.

Goal Akhir (Inference):
Uji AI kamu dengan 2 sensor mesin baru ini:
- Mesin A: Suhu 75, Getaran 30, Suara 65 (Kira-kira masih aman/Normal/Waspada).
- Mesin B: Suhu 140, Getaran 90, Suara 110 (Kira-kira Bahaya/Rusak).

(Hint super penting: Jangan lupa data Mesin A dan B wajib dilewatkan ke rumus Min-Max Scaling menggunakan nilai minimum dan maksimum dari file CSV).

Selamat merakit pabrik AI-mu! Kabari kalau hasil prediksinya sudah keluar. 🏭🔧