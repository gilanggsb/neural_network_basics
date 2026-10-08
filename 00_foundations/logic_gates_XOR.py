import torch
import torch.nn as nn
import torch.optim as optim

# 1. Dataset: Logika XOR
# XOR = Hanya aktif (1) jika salah satu saja yang 1. Jika sama-sama 0 atau sama-sama 1, hasilnya 0.
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


# 2. Arsitektur jaringan (Multi Layer Perceptron)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # LAYER TENGAH (Hidden Layer): 2 Input masuk ke 4 Neuron
        self.hidden = nn.Linear(2, 4)

        # LAYER AKHIR (Output Layer): 4 Neuron disimpulkan menjadi 1 output akhir
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


# panggil modelnya
model = NeuralNetwork()


# 3. Guru & Optimizer
# MSELoss = Mean Squared Error Loss, sama fungsinya untuk melihat seberapa meleset tebakan
criterion = nn.MSELoss()

# Kita pakai Learning Rate 0.1 karena jaringannya sekarang lebih kompleks
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training
epochs = 10000  # Kita butuh epoch lebih banyak karena memecahkan XOR lebih sulit
print("Mulai proses belajar dengan Hidden Layer (ReLU)...\n")

for epoch in range(epochs):
    # Langkah 1: bersihkan sisa - sisa hitungan lama
    optimizer.zero_grad()

    # Langkah 2: suruh model menebak (forward pass)
    output = model(x)

    # Langkah 3: hitung error (Loss)
    loss = criterion(output, y)

    # Langkah 4: AJAIB! pytorch menghitung turunan (derivative) secara OTOMATIS!
    loss.backward()

    # Langkah 5: update bobot dan bias berdasarkan hitungan langkah
    optimizer.step()

    # print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Total Error (Loss): {loss.item():.4f}")

# 5. cek hasil akhir model
print("\n=== HASIL AKHIR ===")
with torch.no_grad():
    # Matikan mode belajar saat ujian
    # mode "jangan hitung turunan lagi" (agar cepat dan efisien)
    predictions = model(x)
    for input_data, output in zip(x, predictions):
        predicted_value = 1.0 if output.item() > 0.5 else 0.0
        print(
            f"Input: {input_data.tolist()} | Tebakan Model: {output.item():.4f} (Dibulatkan: {predicted_value})"
        )
