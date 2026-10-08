import os
import sys

# Tambahkan path folder utama agar bisa import global 'helpers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from helpers.model_handler import load_model, is_model_exists, save_model
import torch
import torch.nn as nn
import torch.optim as optim


# ---- KONSTANTA & KAMUS STATUS ---- #
MODEL_PATH = "factory_machine_monitoring_brain.pth"

STATUS_LABELS = {
    0: "Normal 🟢",
    1: "Waspada 🟡 (Butuh perawatan minggu depan)",
    2: "Bahaya 🟠 (Harus dimatikan hari ini)",
    3: "Rusak Parah 🔴 (Sudah meledak/hancur)",
}


# ---- HELPER FUNCTIONS ---- #
def min_max_scale(tensor, min_vals, max_vals):
    """Menormalisasi fitur tensor ke skala 0.0 - 1.0."""
    return (tensor - min_vals) / (max_vals - min_vals)


def load_dataset(filepath, has_label=True):
    """Membaca file CSV dan mengembalikan tensor fitur (dan label jika has_label=True)."""
    features = []
    labels = []

    with open(filepath, "r") as file:
        lines = file.readlines()[1:]  # Lewati baris header
        for line in lines:
            values = [float(v) for v in line.strip().split(",")]
            if has_label:
                features.append(values[:3])
                labels.append(int(values[3]))
            else:
                features.append(values)

    X = torch.tensor(features, dtype=torch.float32)
    if has_label:
        y = torch.tensor(labels, dtype=torch.long)
        return X, y
    return X


# ---- 1. MEMBACA & MENORMALISASI DATA TRAINING ---- #
X_raw, y = load_dataset("fabric_machine.csv", has_label=True)

# Hitung min dan max tiap fitur langsung dengan PyTorch
features_min = X_raw.min(dim=0).values
features_max = X_raw.max(dim=0).values

print(f"Range Suhu    : {features_min[0].item():.1f} - {features_max[0].item():.1f} °C")
print(f"Range Getaran : {features_min[1].item():.1f} - {features_max[1].item():.1f} Hz")
print(f"Range Suara   : {features_min[2].item():.1f} - {features_max[2].item():.1f} dB")

X_train = min_max_scale(X_raw, features_min, features_max)


# ---- 2. MEMBANGUN ARSITEKTUR MODEL NEURAL NETWORK ---- #
class FactoryMachineMonitoringAI(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 Input (Suhu, Getaran, Suara) -> 8 Hidden -> 4 Output (Status 0-3)
        self.hidden_layer = nn.Linear(3, 8)
        self.output_layer = nn.Linear(8, 4)

    def forward(self, x):
        z1 = torch.relu(self.hidden_layer(x))
        return self.output_layer(z1)


model = FactoryMachineMonitoringAI()
model = load_model(model, MODEL_PATH)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


# ---- 3. TRAINING MODEL ---- #
if not is_model_exists(MODEL_PATH):
    print("\nTraining Model.....")
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
X_test_raw = load_dataset("new_machine.csv", has_label=False)
X_test_scaled = min_max_scale(X_test_raw, features_min, features_max)

with torch.no_grad():
    predictions = model(X_test_scaled)
    predicted_classes = torch.argmax(predictions, dim=1)

    print("\n=== HASIL PEMANTAUAN MESIN BARU ===")
    for i in range(len(X_test_raw)):
        suhu = X_test_raw[i][0].item()
        getaran = X_test_raw[i][1].item()
        suara = X_test_raw[i][2].item()
        status = STATUS_LABELS[int(predicted_classes[i].item())]

        print(
            f"Machine {i + 1}:\n"
            f"  Suhu   : {suhu:.1f} °C\n"
            f"  Getaran: {getaran:.1f} Hz\n"
            f"  Suara  : {suara:.1f} dB\n"
            f"  Status : {status}\n"
        )
