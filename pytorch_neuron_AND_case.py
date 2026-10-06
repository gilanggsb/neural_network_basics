import torch
import torch.nn as nn
import torch.optim as optim

# 1. Siapkan data logika AND menggunakan Tensor
# tipe datanya harus float agar bisa dihitung errornya
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

# Target jawaban benar
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])


# 2. bikin model neuron
class PyTorchNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        """ nn.Linear(2, 1) artinya: 2 Input, 1 Output.
        AJAIBNYA: PyTorch otomatis membuatkan Weights dan Bias acak di balik layar!"""
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        """Mirip fungsi forward kita kemarin: (Input * Weights) + Bias, lalu di-Sigmoid"""
        z = self.layer(x)
        return torch.sigmoid(z)


# panggil modelnya
model = PyTorchNeuron()

# 3. Loss cara menghitung error
# MSELoss = Mean Squared Error Loss, sama fungsinya untuk melihat seberapa meleset tebakan
criterion = nn.MSELoss()

# Optimizer (cara mengkoreksi bobot / bias)
# SGD = Stochastic Gradient Descent, metode koreksi paling klasik. yang akan melakukan update bobot
# lr = learning_rate (seberapa cepat belajar)
optimizer = optim.SGD(model.parameters(), lr=0.5)


# 4. mulai training
epochs = 5000
print(f"Mulai proses training dengan pytorch...\n")

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


# 5. cek hasil akhir model sudah belajar belum?
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
