import os
import sys

# Tambahkan path folder utama agar bisa import global 'helpers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from helpers.model_handler import save_model, load_model, is_model_exists
import torch
import torch.nn as nn
import torch.optim as optim


# ---- helper function ----
otak_fake_news_detector_ai_path = "models/otak_fake_news_detector_ai.pth"


def extract_data(data):
    kalimat = data[0]
    label = None
    if len(data) > 1 and data[1]:
        label = float(data[1])

    return kalimat, label


def sentence_to_vector(sentence, vocab):
    text_vector = [0.0] * len(vocab)
    for i in range(len(vocab)):
        keyword = vocab[i]
        if keyword in sentence:
            text_vector[i] = 1.0

    return text_vector


# Kamus standar kita (Panjang = 8 kata)
vocab = [
    "pemerintah",
    "vaksin",
    "gratis",
    "aman",
    "bahaya",
    "chip",
    "pelacak",
    "rahasia",
]
data_features = []
data_targets = []
# ---- 1.Membaca Data Dari Text ---- #
with open("dataset_berita.csv", "r") as file:
    lines = file.readlines()[1:]  # Skip header
    for line in lines:
        kolom = line.strip().split(",")  # pecah teks berdasarkan koma
        kalimat, label = extract_data(kolom)
        # print(f"cek label {label} | kalimat {kalimat} | kolom {kolom}")

        # # --- mulai logika BAG-OF-Words ---
        # # 1. Siapkan array berisi angka 0.0 sepanjang jumlah vocab (8 buah)
        # vektor_teks = [0.0] * len(vocab)

        # # 2. Cek setiap kata di dalam vocab satu per satu
        # for i in range(len(vocab)):
        #     kata_kunci = vocab[i]

        #     # 3. cek apakah kata kunci tersebut ada di dalam kalimat
        #     # HINT: gunakan 'in' di Python
        #     # Tulis logika IF-mu di sini:
        #     # Jika kata_kunci ada di dalam kalimat:
        #     #     ubah vektor_teks pada index [i] menjadi 1.0

        #     # --- Tulis Logika IF di sini ---
        #     if kata_kunci in kalimat:
        #         vektor_teks[i] = 1.0

        # Simpan vektor_teks yang sudah terisi angka 1 dan 0 ke data_features
        text_vector = sentence_to_vector(kalimat, vocab)
        data_features.append(text_vector)
        data_targets.append([label])
        # Ingat, target untuk Binary/BCELoss butuh float, cth: [1.0]

# Terakhir, ubah list Python menjadi Tensor PyTorch
X = torch.tensor(data_features, dtype=torch.float32)
y = torch.tensor(data_targets, dtype=torch.float32)

# Kalau mau ngecek hasilnya bener atau ngga, coba print data pertama:
print("Kalimat aslinya:", lines[0].strip())
print("Hasil vektornya:", X[0].tolist())


# ---- 3. Definisi Model Neural Network ---- #
class FakeNewsDetectorAI(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 Input(vocab) -> 16 neuron
        self.hidden = nn.Linear(8, 16)
        # 16 Neuron -> 1 Output (Keluarkan 1 angka persentase)
        self.output = nn.Linear(16, 1)

    def forward(self, X):
        z1 = torch.relu(self.hidden(X))
        return torch.sigmoid(self.output(z1))


model = FakeNewsDetectorAI()
model = load_model(model, otak_fake_news_detector_ai_path)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


if not is_model_exists(otak_fake_news_detector_ai_path):
    # ---- 4. TRAINING AI ---- #
    print("\nMulai training AI Fake News Detector...")
    epochs = 2000
    for epoch in range(epochs):
        # Langkah 1: bersihkan sisa - sisa hitungan lama
        optimizer.zero_grad()

        # Langkah 2: suruh model menebak (forward pass)
        prediction = model(X)

        # Langkah 3: hitung error (loss)
        loss = criterion(prediction, y)

        # Langkah 4: menghitung turunan (derivative) secara OTOMATIS
        loss.backward()

        # Langkah 5: update bobot dan bias berdasarkan hitungan langkah
        optimizer.step()

        # print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Total Error (Loss): {loss.item():.4f}")


# ---- 5. UJIAN AKHIR (INFERENCE) ---- #
teks_uji_1 = "pemerintah berikan vaksin gratis yang aman"
teks_uji_1_vector = sentence_to_vector(teks_uji_1, vocab)
print(f"cek teks uji 1: {teks_uji_1_vector}")

teks_uji_2 = "awas bahaya rahasia chip pelacak"
teks_uji_2_vector = sentence_to_vector(teks_uji_2, vocab)
print(f"cek teks uji 2: {teks_uji_2_vector}")

news_vector = []
news_vector.append(teks_uji_1_vector)
news_vector.append(teks_uji_2_vector)

with torch.no_grad():
    tensor_news = torch.tensor(news_vector, dtype=torch.float32)

    prediction = model(tensor_news)
    print(f"Hasil Prediksi: {prediction}")

    for i in range(len(news_vector)):
        # Karena Sigmoid, hasil berupa Float (misal 0.85)
        prediction_result = prediction[i].item()

        # Jika hasil mendekati 1 (di atas 0.5) maka Hoaks, sisanya Asli
        news_label = "FAKE NEWS" if prediction_result > 0.5 else "REAL NEWS"

        # Output
        print(
            f"\nHasil Prediksi Berita {i+1}: {news_label} (Kepercayaan: {prediction_result:.2f})"
        )

# simpan model
save_model(model, otak_fake_news_detector_ai_path)
