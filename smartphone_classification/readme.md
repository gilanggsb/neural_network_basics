Ini dia! Kita akan masuk ke ranah Data Preprocessing (Persiapan Data). Di dunia nyata, 80% waktu seorang AI Engineer dihabiskan di tahap ini.

Neural Network itu "benci" dengan angka yang jaraknya terlalu jomplang. Kalau kamu memasukkan angka Baterai 5000 dan Harga 2.5 secara bersamaan, angka 5000 akan "menelan" angka 2.5 saat dikalikan bobot. Hasilnya, AI jadi bias dan susah belajar. 

Solusinya? Kita harus menjinakkan data tersebut menggunakan teknik Min-Max Scaling (Normalisasi), yaitu memampatkan rentang asli berapapun menjadi 0.0 sampai 1.0.

---

Skenario: AI Pengklasifikasi Kelas Smartphone 📱
Kita akan buat AI yang memprediksi apakah sebuah HP masuk kelas:
- 0 = Entry-Level (Murah)
- 1 = Mid-Range (Menengah)
- 2 = Flagship (Premium)

Fitur aslinya (sangat beragam rentangnya):
1. Kapasitas Baterai (Ribuan: 3000 - 6000 mAh)
2. RAM (Satuan: 2 - 12 GB)
3. Harga (Jutaan: 1.5 - 20.0 Juta Rp)