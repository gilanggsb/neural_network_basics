import os
import sys

# Tambahkan path folder utama agar bisa import global 'helpers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from helpers.model_handler import load_model, is_model_exists, save_model
import torch
import torch.nn as nn
import torch.optim as optim

# --- KONSTANTA & KAMUS STATUS ---- #
MODEL_PATH = "customer_churn_brain.pth"

CHURN_LABELS = {0: "Bertahan", 1: "Kabur"}

CONTRACT_TYPE_DICT = {"bulanan": 0.0, "tahunan": 1.0}


# ---- HELPER FUNCTIONS ---- #
def is_digit(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def min_max_scale(tensor, min_vals, max_vals):
    return (tensor - min_vals) / (max_vals - min_vals)


def load_dataset(filepath, has_label=True):
    features = []
    labels = []

    with open(filepath, "r") as file:
        next(file)  # Skip baris header pertama
        for line in file:
            values = [
                CONTRACT_TYPE_DICT[i.lower()] if not is_digit(i) else float(i)
                for i in line.strip().split(",")
            ]
            if not has_label:
                features.append(values)
                continue

            features.append(values[:3])
            labels.append(int(values[3]))

    X = torch.tensor(features, dtype=torch.float32)
    y = None if not has_label else torch.tensor(labels, dtype=torch.long)
    return X, y


# ---- 1. MEMBACA & MENORMALISASI DATA TRAINING ---- #
X_raw, y = load_dataset("telekom_churn.csv")

# hitung min dan max tiap fitur langsung dengan pyTorch
features_min = X_raw.min(dim=0).values
features_max = X_raw.max(dim=0).values

# 2. FIX: Paksa nilai min=0 dan max=1 untuk kolom "Tipe_Kontrak" (index 2)# agar rumus (X - 0) / (1 - 0) menghasilkan angka aslinya kembali.
features_min[2] = 0.0
features_max[2] = 1.0

print(
    f"Range Bulan Langganan: {features_min[0].item():.1f} - {features_max[0].item():.1f}"
)
print(
    f"Range Tagihan Bulanan: {features_min[1].item():.1f} - {features_max[1].item():.1f}"
)
print(
    f"Range Tipe Kontrak Bulanan(0) atau Tahunan(1): {features_min[2].item():.1f} - {features_max[2].item():.1f}"
)

X_train = min_max_scale(X_raw, features_min, features_max)


# ---- 2. MEMBANGUN ARSITEKTUR MODEL NEURAL NETWORK ---- #
class CustomerChurnNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(3, 8)
        self.output_layer = nn.Linear(8, 2)

    def forward(self, X):
        z1 = torch.relu(self.hidden_layer(X))
        return self.output_layer(z1)


model = CustomerChurnNN()
model = load_model(model, MODEL_PATH)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# ---- 3. TRAINING MODEL ---- #
if not is_model_exists(MODEL_PATH):
    print(f"\nTraining model...")
    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    save_model(model, MODEL_PATH)


# ---- 4. INFERENCE (PENGUJIAN MESIN BARU) ---- #
X_test_raw, _ = load_dataset("new_customer_churn.csv", has_label=False)
X_test_scaled = min_max_scale(X_test_raw, features_min, features_max)

with torch.no_grad():
    predictions = model(X_test_scaled)
    prediction_classes = torch.argmax(predictions, dim=1)

    print("\n=== HASIL PREDIKSI CUSTOMER BARU ===")
    for i in range(len(X_test_raw)):
        monthly_subscribe = X_test_raw[i][0].item()
        monthly_paid = X_test_raw[i][1].item()
        subscribe_type = X_test_raw[i][2].item()
        churn_prediction = CHURN_LABELS[int(prediction_classes[i].item())]

        print(
            f"Churn {i + 1}:\n"
            f"  Monthly subs   : {monthly_subscribe}\n"
            f"  Monthly paid: {monthly_paid}\n"
            f"  Subs type  : {subscribe_type}\n"
            f"  Churn : {churn_prediction}\n"
        )
