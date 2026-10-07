import torch
import torch.nn as nn
import torch.optim as optim


# ---- helper function ---- #
def extract_data(data: list):
    baterai = float(data[0])
    ram = float(data[1])
    harga = float(data[2])

    kelas = None
    if len(data) > 3 and data[3]:
        kelas = int(data[3])

    return baterai, ram, harga, kelas


def calculate_min_max(data, min, max):
    return (data - min) / (max - min)


# ---- 1.Membaca Data Dari Text ---- #
data_features = []
data_targets = []

print("Membaca file dataset_hp.csv...")
with open("dataset_hp.csv", "r") as file:
    lines = file.readlines()[1:]  # Skip baris pertama (Header)
    for line in lines:
        kolom = line.strip().split(",")  # pecah teks berdasarkan koma

        baterai, ram, harga, kelas = extract_data(kolom)
        data_features.append([baterai, ram, harga])
        data_targets.append(kelas)

# ---- 2. Normalisasi Data (MIN - MAX SCALING) ---- #
# Rumus = (Nilai - Nilai_Minimum) / (Nilai_Maksimum - Nilai_Minimum)
# Ekstrak tiap kolom untuk mencari nilai paling kecil (min) dan paling besar (max)
baterai_all = [row[0] for row in data_features]
ram_all = [row[1] for row in data_features]
harga_all = [row[2] for row in data_features]

min_bat, max_bat = min(baterai_all), max(baterai_all)
min_ram, max_ram = min(ram_all), max(ram_all)
min_harga, max_harga = min(harga_all), max(harga_all)

print(f"Rentang Baterai: {min_bat} - {max_bat}")
print(f"Rentang RAM: {min_ram} - {max_ram}")
print(f"Rentang Harga: {min_harga} - {max_harga}")

# Terapkan rumusnya ke seluruh baris data
features_normalized = []
for row in data_features:
    norm_bat = calculate_min_max(row[0], min_bat, max_bat)
    norm_ram = calculate_min_max(row[1], min_ram, max_ram)
    norm_harga = calculate_min_max(row[2], min_harga, max_harga)
    features_normalized.append([norm_bat, norm_ram, norm_harga])

# Ubah List Python menjadi Tensor PyTorch
x = torch.tensor(features_normalized, dtype=torch.float32)
y = torch.tensor(data_targets, dtype=torch.long)


# --- 3. BIKIN MODEL AI (MULTI-CLASS) ---
class SmartPhoneAI(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 Input(Baterai, RAM, Harga) -> 8 Neuron
        self.hidden = nn.Linear(3, 8)
        # 8 Neuron -> 3 Output(Entry, Mid, Flagship)
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        z1 = torch.relu(self.hidden(x))
        return self.output(z1)  # tanpa sigmoid (pakai CrossEntropy)


model = SmartPhoneAI()
model.load_state_dict(torch.load("otak_hp_ai.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# # ---- 4. TRAINING AI ---- #
# epochs = 10000
# print("\nMulai training AI Smartphone...")
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     y_pred = model(x)
#     loss = criterion(y_pred, y)
#     loss.backward()
#     optimizer.step()

#     # print progress
#     if epoch % 1000 == 0:
#         print(f"Epoch {epoch} | Total Error (Loss): {loss.item():.4f}")


# --- 5. UJIAN: MASUKKAN HP BARU MENTAH (BELUM DINORMALISASI) ---
print("\n=== TEBAK HP BARU ===")
# Data Mentah: [Baterai, RAM, Harga]
hp_baru_mentah = [
    [4500, 4, 2.0],  # HP A (Harusnya Entry-Level)
    [5000, 8, 6.0],  # HP B (Harusnya Mid-Range)
    [5000, 8, 6.0],  # HP B (Harusnya Mid-Range)
    [4000, 12, 18.0],  # HP C (Harusnya Flagship. Baterai kecil tapi RAM & Harga sultan)
]


kamus_kelas = {0: "Entry-Level 🥉", 1: "Mid-Range 🥈", 2: "Flagship 🥇"}
with torch.no_grad():
    # KUNCI PENTING: Data baru juga WAJIB dinormalisasi pakai skala (Min/Max) yang lama!
    hp_baru_norm = []
    for row in hp_baru_mentah:
        baterai, ram, harga, kelas = extract_data(row)
        n_baterai = calculate_min_max(baterai, min_bat, max_bat)
        n_ram = calculate_min_max(ram, min_ram, max_ram)
        n_harga = calculate_min_max(harga, min_harga, max_harga)

        hp_baru_norm.append([n_baterai, n_ram, n_harga])

    tensor_hp_baru = torch.tensor(hp_baru_norm, dtype=torch.float32)

    # AI menebak
    prediksi = model(tensor_hp_baru)
    print(f"Hasil Prediksi: {prediksi}")
    keputusan = torch.argmax(prediksi, dim=1)
    print(f"Keputusan: {keputusan}")
    for i in range(len(hp_baru_mentah)):
        spek = hp_baru_mentah[i]
        tebakan = kamus_kelas[int(keputusan[i].item())]

        print(
            f"HP (Bat: {spek[0]}mAh, RAM: {spek[1]}GB, Harga: {spek[2]}jt) -> {tebakan}"
        )

# simpan model
torch.save(model.state_dict(), "otak_hp_ai.pth")
