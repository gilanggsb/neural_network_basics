import math
import random


class SimpleNeuron:
    def __init__(self, num_inputs):
        "1. inisialisasi bobot (weights) dan bias dari angka random"
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def sigmoid(self, x):
        "2. Fungsi Aktivasi: Sigmoid (mengubah output ke rentang 0.0 sampai 1.0)"
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        "pastikan jumlah input sesuai dengan jumlah bobot"
        if len(inputs) != len(self.weights):
            raise ValueError("Jumlah input harus sesuai dengan jumlah bobot")

        # 3. hitung total (input x bobot) + bias
        z = sum(i * w for i, w in zip(inputs, self.weights)) + self.bias

        # 4. lewatkan fungsi aktivasi
        return self.sigmoid(z)

    def sigmoid_derivative(self, output):
        """
        ini adalah rumus turunan matematika dari fungsi sigmoid
        fungsinya untuk mencari tau "seberapa curam kemiringan tebakan kita?"
        semakin curam, semakin besar pergeseran bobot yang harus kita lakukan
        """
        return output * (1 - output)

    def train(self, inputs, target, learning_rate=0.1):
        """
        ini adalah proses "belajar" nya si neuron menggunakan backpropagation
        """
        # 1. suruh neuron menebak dulu pake bobot yang saat ini (Forward pass)
        output = self.forward(inputs)

        # 2. hitung selisih antara jawaban benar (target) dengan tebakan neuron
        error = target - output

        # 3. hitung seberapa besar koreksi yang harus dilakukan
        # Rumusnya: selisih (error) dikali tingkat kecuraman kurva (derivative)
        adjustment = error * self.sigmoid_derivative(output)

        # 4. saatnya mengkoreksi tiap bobot (weights)
        for i in range(len(self.weights)):
            # boobt baru = bobot lama + (kecepatan dasar * koreksi * data input asli)
            self.weights[i] += learning_rate * adjustment * inputs[i]

        # 5. jangan lupa koreksi biasnya juga
        # bias tidak dikali input karena dia berdiri sendiri (bawaan sifat)
        self.bias += learning_rate * adjustment

        # kembalikan nilai error supaya kita bisa pantau apakah neuron makin pintar
        return error


# --- UJI COBA ---

# Buat satu neuron yang menerima 3 input
neuron = SimpleNeuron(num_inputs=3)

# Mock Data (contoh: sensor cuaca, suhu, kelembapan)
data_input = [0.8, 0.2, 0.9]

# Jalankan forward pass (proses menebak/kalkulasi)
output = neuron.forward(data_input)

# Jalankan forward pass (proses menebak / kalkulasi)
print("\n--- Hasil Proses Neuron ---")
print(f"Data Input : {data_input}")
print(f"Bobot awal :  {[round(w,4) for w in neuron.weights]}")
print(f"Bias {neuron.bias:.4f}")
print(f"Output.    : {output:.4f} (mendekati 1 = Aktif, mendekati 0 = Tidak)")
print("--- Selesai ---\n")


# --- UJI COBA Neuron Back Propagation ---

# Buat satu neuron baru yang menerima 2 input
neuron = SimpleNeuron(num_inputs=2)

# Siapkan soal ujian (Kita ajari logika 'AND')# Neuron hanya boleh menjawab 1 (Aktif) jika KEDUA inputnya 1.
training_data = [
    ([0, 0], 0),  # Input 0, 0 -> Harusnya jawab 0
    ([0, 1], 0),  # Input 0, 1 -> Harusnya jawab 0
    ([1, 0], 0),  # Input 1, 0 -> Harusnya jawab 0
    ([1, 1], 1),  # Input 1, 1 -> Harusnya jawab 1
]

# kita suruh neuron belajar soal yang sama diulang 5000 kali
epochs = 5000

print("\n--- Memulai Pelatihan Neuron (Backpropagation) ---")
for epoch in range(epochs):
    total_error = 0

    # berikan 4 soal diatas satu per satu
    for inputs, target in training_data:
        # panggil fungsi train yang telah dibuat
        error = neuron.train(inputs, target, learning_rate=0.5)

        # kumpulkan total error untuk melihat perkembangannya
        total_error += error

    # Cetak progres setiap 1000 kali putaran
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs}, Total Error: {total_error:.4f}")


print("\n--- Hasil Akhir Pelatihan ---")
for inputs, target in training_data:
    hasil = neuron.forward(inputs)
    # Kita bulatkan hasil akhirnya supaya jadi angka tegas 0 atau 1
    print(
        f"Input: {inputs} | Target: {target} | Jawaban Neuron: {hasil:.4f} -> {round(hasil)}"
    )
