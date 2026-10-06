# Skenarionya:
# Sebuah bank ingin AI yang bisa memutuskan apakah pengajuan pinjaman/kredit nasabah Disetujui (1) atau Ditolak (0).
# Kita akan pakai 2 faktor (Input):
# 1. Pendapatan per bulan (Kita skalakan 0.1 sampai 1.0. Misal 0.5 artinya 5 Juta, 1.0 artinya 10 Juta).
# 2. Jumlah Utang saat ini (Sama, diskalakan 0.1 sampai 1.0).

# Logika manusianya begini:
# - Gaji besar, utang kecil 👉 Pasti Disetujui (1)
# - Gaji kecil, utang besar 👉 Pasti Ditolak (0)
# - Gaji sedang, utang sedang 👉 Tergantung (AI harus belajar polanya)


import torch
import torch.nn as nn
import torch.optim as optim


# 1. Dataset Nasabah (Data sejarah bank)
# Format: [Pendapatan, Jumlah Utang] (Skala 1-10, yang diubah jadi 0.1-1.0)
x = torch.tensor(
    [
        [0.8, 0.1],  # Nasabah A: Gaji 8jt, Utang 1jt
        [0.3, 0.0],  # Nasabah B: Gaji 3jt, Utang 0jt (Gaji kecil tapi bebas utang)
        [0.9, 0.7],  # Nasabah C: Gaji 9jt, Utang 7jt (Gaji besar, sanggup bayar utang)
        [0.2, 0.8],  # Nasabah D: Gaji 2jt, Utang 8jt
        [0.5, 0.6],  # Nasabah E: Gaji 5jt, Utang 6jt
        [0.4, 0.2],  # Nasabah F: Gaji 4jt, Utang 2jt
    ]
)

# Target: 1 (Setuju), 0 (Tolak)
y = torch.tensor(
    [
        [1.0],  # A: Setuju
        [1.0],  # B: Setuju
        [1.0],  # C: Setuju
        [0.0],  # D: Tolak (Utang terlalu besar untuk gajinya)
        [0.0],  # E: Tolak (Utang lebih besar dari gaji)
        [1.0],  # F: Setuju (Utang masih wajar)
    ]
)


# 2. Arsitektur Jaringan
class LoanApprovalAi(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 Input (Gaji, Utang) -> 4 Neuron (Hidden Layer)
        self.hidden = nn.Linear(2, 4)

        # 4 Neuron -> 1 Keputusan (Output)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        # Tahap 1: Data masuk ke hidden layer
        z1 = self.hidden(x)
        # Gunakan ReLU di layer tengah!
        a1 = torch.relu(z1)

        # Tahap 2: hasil dari layer tengah masuk ke layer akhir
        z2 = self.output(a1)
        # Gunakan sigmoid di layer akhir (karena butuh probabilitas 0-1)
        a2 = torch.sigmoid(z2)

        return a2


# panggil model
model = LoanApprovalAi()

# 3. Guru & Optimizer
# MSELoss = Mean Squared Error Loss, sama fungsinya untuk melihat seberapa meleset tebakan
criterion = nn.MSELoss()

# Kita pakai Learning Rate 0.1 karena jaringannya sekarang lebih kompleks
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training
epochs = 5000
print("Mulai proses training dengan Hidden Layer (ReLU)...\n")


for epoch in range(epochs):
    # Langkah 1: bersihkan sisa - sisa hitungan lama
    optimizer.zero_grad()

    # Langkah 2: suruh model menebak (forward pass)
    output = model(x)

    # Langkah 3: hitung error (Loss)
    loss = criterion(output, y)

    # Langkah 4: menghitung turunan (derivative) secara OTOMATIS!
    loss.backward()

    # Langkah 5: update bobot dan bias berdasarkan hitungan langkah
    optimizer.step()

    # print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Total Error (Loss): {loss.item():.4f}")


# 5. UJIAN AKHIR: PREDIKSI NASABAH BARU!print("\n=== KEDATANGAN 3 NASABAH BARU ===")
# Data nasabah baru yang BELUM PERNAH dilihat oleh AI saat belajar
nasabah_baru = torch.tensor(
    [
        [0.7, 0.2],  # Nasabah X: Gaji 7jt, Utang 2jt
        [0.3, 0.7],  # Nasabah Y: Gaji 3jt, Utang 7jt
        [0.6, 0.5],  # Nasabah Z: Gaji 6jt, Utang 5jt (Kasus meragukan / mepet)
    ]
)

with torch.no_grad():
    hasil_prediksi = model(nasabah_baru)
    print(f"hasil prediksi = {hasil_prediksi}")
    print("Format input: [pendapatan, utang]")
    for i in range(len(nasabah_baru)):
        print(
            f"cek data {i} =  nasabah_baru[i] -> {nasabah_baru[i][0]} || nasabah_baru[i][1] -> {nasabah_baru[i][1]} || nasabah_baru[i][0].item() -> {nasabah_baru[i][0].item()}"
        )
        gaji = nasabah_baru[i][0].item() * 10
        utang = nasabah_baru[i][1].item() * 10
        skor = hasil_prediksi[i].item()

        status = "DISETUJUI ✅" if skor >= 0.5 else "DITOLAK ❌"

        print(
            f"Nasabah {i+1} (Gaji: {gaji:.1f}jt, Utang: {utang:.1f}jt) -> Skor AI: {skor:.2f} ({status})"
        )
