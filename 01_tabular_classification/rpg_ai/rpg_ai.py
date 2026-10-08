# Kalau sebelumnya kita terus-terusan berkutat di Klasifikasi Biner (jawabannya cuma 2: Setuju/Tolak, Nyala/Mati, 0/1), sekarang kita akan "naik level" ke masalah yang lebih kompleks: Multi-Class Classification (Memilih 1 dari banyak pilihan).

# Skenarionya: AI Penentu Role Game RPG 🗡️🧙‍♂️🏹
# Bayangkan kamu bikin game. Saat pemain baru mendaftar, mereka disuruh tes fisik (Kekuatan) dan tes sihir (Kecerdasan). Berdasarkan 2 nilai ini, AI harus menentukan apakah pemain ini cocok jadi Warrior (0), Mage (1), atau Archer (2).

# Aturan mainnya (logika manusia):
# - Otot besar, Sihir lemah 👉 Warrior (Class 0)
# - Otot lemah, Sihir kuat 👉 Mage (Class 1)
# - Otot seimbang, Sihir seimbang 👉 Archer (Class 2)

# Hal Baru yang Akan Kita Pelajari:
# 1. Output Neuronnya tidak lagi 1! Karena ada 3 kemungkinan role, Output Layer kita harus punya 3 Neuron.
# 2. Selamat tinggal Sigmoid & MSELoss! Untuk tebakan multi-pilihan, kita pakai fungsi Loss yang jauh lebih sakti bernama CrossEntropyLoss.
from torch.distributed import init_device_mesh
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Dataset Pemain
# Format: [Kekuatan Fisik, Kekuatan Sihir] (Skala 0.1 - 1.0)
x = torch.tensor(
    [
        [0.9, 0.1],  # Pemain A: Fisik 90, Sihir 10
        [0.1, 0.9],  # Pemain B: Fisik 10, Sihir 90
        [0.5, 0.5],  # Pemain C: Fisik 50, Sihir 50
        [0.8, 0.2],  # Pemain D: Fisik 80, Sihir 20
        [0.2, 0.8],  # Pemain E: Fisik 20, Sihir 80
        [0.6, 0.6],  # Pemain F: Fisik 60, Sihir 60
    ]
)

# Target Role: 0 (Warrior), 1 (Mage), 2 (Archer)
# # PERHATIKAN: Tipe datanya bukan float (koma) lagi, tapi Integer (Long)!
y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)


# 2. Arsitektur jaringan Multi-Class
class RPGClassifierAI(nn.Module):
    def __init__(self):
        super().__init__()
        # Hidden Layer: 2 Input -> 8 Neuron (Diperbesar supaya makin pintar)
        self.hidden = nn.Linear(2, 8)
        # Output Layer: 8 Neuron -> 3 Output (Karena kita punya 3 Role!)
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        # Tahap 1: Data masuk ke hidden layer
        z1 = self.hidden(x)
        # Gunakan ReLU di layer tengah!
        a1 = torch.relu(z1)

        # Tahap 2: hasil dari layer tengah masuk ke layer akhir
        z2 = self.output(a1)

        # PERHATIKAN: Tidak ada Sigmoid di sini!
        # Biarkan angkanya mentah (disebut Logits).
        return z2


model = RPGClassifierAI()
model.load_state_dict(torch.load("./rpg_ai/otak_rpg_ai.pth"))

# 3. Guru & Optimizer (Khusus untuk Multi-Class)
# CrossEntropyLoss otomatis mengubah angka mentah (Logits) menjadi persentase probabilitas (Softmax)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training (Multi-Class)
# epochs = 10000
# print("AI sedang menyeleksi kandidat RPG...\n")

# for epoch in range(epochs):
#     # Langkah 1: bersihkan sisa - sisa hitungan lama
#     optimizer.zero_grad()

#     # Langkah 2: suruh model menebak (forward pass)
#     predictions = model(x)

#     # Langkah 3: hitung error (loss)
#     loss = criterion(predictions, y)

#     # Langkah 4: menghitung turunan (derivative) secara OTOMATIS!
#     loss.backward()

#     # Langkah 5: update bobot dan bias berdasarkan hitungan langkah
#     optimizer.step()

#     # print progress
#     if epoch % 1000 == 0:
#         print(f"Epoch {epoch} | Total Error (Loss): {loss.item():.4f}")


# 5. UJIAN AKHIR: Pemain Baru Login!print("\n=== PEMAIN BARU LOGIN ===")
pemain_baru = torch.tensor(
    [
        [0.85, 0.15],  # Harusnya Warrior
        [0.15, 0.85],  # Harusnya Mage
        [0.55, 0.45],  # Harusnya Archer
        [0.35, 0.69],  # Harusnya Mage
        [0.70, 0.80],  # Tipe aneh (Fisik kuat, Sihir kuat) -> Kita lihat AI milih apa!
    ]
)

# Kamus untuk menerjemahkan angka ke nama Role
role_dict = {0: "Warrior 🗡️", 1: "Mage 🧙‍♂️", 2: "Archer 🏹"}

model.eval()
with torch.no_grad():
    # Hasil tebakan masih berupa 3 angka mentah untuk setiap pemain
    hasil_mentah = model(pemain_baru)

    # Kita pakai torch.argmax untuk mencari neuron mana yang nilainya paling tinggi
    # dim=1 artinya cari yang tertinggi di setiap baris
    keputusan_akhir = torch.argmax(hasil_mentah, dim=1)

    # Tampilkan hasilnya
    print("\nHasil Seleksi Role Pemain Baru:")
    print("--------------------------------")
    for i, keputusan in enumerate(keputusan_akhir):
        fisik = pemain_baru[i][0].item() * 100
        sihir = pemain_baru[i][1].item() * 100

        # Ambil index juara (0, 1, atau 2)
        index_juara = int(keputusan_akhir[i].item())
        role_terpilih = role_dict[index_juara]

        print(
            f"Pemain {i+1} (Fisik: {fisik:.0f}, Sihir: {sihir:.0f}) -> AI Memilih: {role_terpilih}"
        )

# simpan model
# torch.save(model.state_dict(), "./rpg_ai/otak_rpg_ai.pth")
