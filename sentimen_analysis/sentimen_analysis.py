import os
import sys

# Tambahkan path folder utama agar bisa import global 'helpers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helpers.model_handler import load_model, is_model_exists, save_model
import torch
import torch.nn as nn
import torch.optim as optim


vocab = [
    "jelek",
    "rusak",
    "parah",
    "cacat",
    "kecewa",
    "hancur",
    "lumayan",
    "sesuai",
    "biasa",
    "standar",
    "bagus",
    "memuaskan",
    "mantap",
    "cepat",
    "sempurna",
    "suka",
]


# --- helper function ----
otak_sentimen_ai_path = "otak_sentimen_ai_path"


def extract_data(data):
    kalimat = data[0]
    label = None
    if len(data) > 1 and data[1]:
        label = int(data[1])
    return kalimat, label


def sentence_to_vector(sentence: str):
    words = sentence.lower().split()
    vector = [0.0] * len(vocab)

    for i, word in enumerate(vocab):
        if word in words:
            vector[i] = 1.0
    return vector


data_features = []
data_targets = []
# ---- 1. MEMBACA DATA & Normalisasi DATA ---- #
with open("dataset_review.csv", "r") as file:
    lines = file.readlines()[1:]  # Skip header
    for line in lines:
        kolom = line.strip().split(",")

        sentence, label = extract_data(kolom)
        text_vector = sentence_to_vector(sentence)

        data_features.append(text_vector)
        data_targets.append(label)


# ubah list python menjadi tensor pytorch
X = torch.tensor(data_features, dtype=torch.float)
y = torch.tensor(data_targets, dtype=torch.long)

# print("Kalimat aslinya:", lines[0].strip())
# print("Hasil vektornya:", X[0].tolist())


# ---- 2. MEMBANGUN JARINGAN NEURAL ---- #
class SentimenAnalysisAI(nn.Module):
    def __init__(self):
        super().__init__()
        # len(vocab) Input -> 32 Neuron -> 3 Output
        self.hidden = nn.Linear(len(vocab), 32)
        self.output = nn.Linear(32, 3)

    def forward(self, X):
        z1 = torch.relu(self.hidden(X))
        return self.output(z1)


model = SentimenAnalysisAI()
model = load_model(model, otak_sentimen_ai_path)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- 4. TRAINING ---- #
if not is_model_exists(otak_sentimen_ai_path):
    print("Training model... This may take a few minutes...")
    epochs = 10000
    for epoch in range(epochs):
        # Langkah 1: bersihkan sisa - sisa hitunga lama
        optimizer.zero_grad()

        # Langkah 2: prediksi data (forward pass)
        prediction = model(X)

        # Langkah 3: hitung loss
        loss = criterion(prediction, y)

        # Langkah 4: hitung turunan (derivative) dari loss
        loss.backward()

        # Langkah 5: update bobot & bias berdasarkan step
        optimizer.step()

        # print loss setiap 1000 epoch
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# --- 5. INFERENCE (UJI COBA) ---
review_uji_1 = "barang jelek rusak parah tidak sesuai"
review_uji_2 = "pengiriman lumayan tapi barangnya jelek rusak"
review_uji_3 = "barang sempurna memuaskan"

review_uji_1_vector = sentence_to_vector(review_uji_1)
review_uji_2_vector = sentence_to_vector(review_uji_2)
review_uji_3_vector = sentence_to_vector(review_uji_3)

sentiment_list = [review_uji_1, review_uji_2, review_uji_3]
sentimen_vector = [review_uji_1_vector, review_uji_2_vector, review_uji_3_vector]

kamus_sentimen = {0: "Negatif 😡", 1: "Netral 😐", 2: "Positif 😊"}
with torch.no_grad():
    tensor_sentimen = torch.tensor(sentimen_vector, dtype=torch.float)

    prediction = model(tensor_sentimen)
    # print(f"Hasil prediksi: {prediction}")

    decision_threshold = torch.argmax(prediction, dim=1)
    # print(f"Keputusan threshold: {decision_threshold}")

    for i in range(len(sentimen_vector)):
        prediction_result = int(decision_threshold[i].item())
        print(
            f"Ulasan : '{sentiment_list[i]}'\nSentimen : {kamus_sentimen[prediction_result]}\n"
        )


save_model(model, otak_sentimen_ai_path)
