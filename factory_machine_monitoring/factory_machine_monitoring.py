import os
import sys

# Tambahkan path folder utama agar bisa import global 'helpers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helpers.model_handler import load_model, is_model_exists, save_model
import torch
import torch.nn as nn
import torch.optim as optim


# ---- helper function ---- #
factory_machine_monitoring_brain_path = "factory_machine_monitoring_brain.pth"


def extract_data(data):
    suhu = float(data[0])
    getaran = float(data[1])
    suara = float(data[2])
    status = None
    if len(data) > 3 and data[3]:
        status = int(data[3])

    return suhu, getaran, suara, status


def calculate_min_max(value, min, max):
    return (value - min) / (max - min)


def normalize_features(features, suhu_minmax, getaran_minmax, suara_minmax):
    features_normalized = []
    for row in features:
        n_suhu = calculate_min_max(row[0], suhu_minmax[0], suhu_minmax[1])
        n_getaran = calculate_min_max(row[1], getaran_minmax[0], getaran_minmax[1])
        n_suara = calculate_min_max(row[2], suara_minmax[0], suara_minmax[1])
        features_normalized.append([n_suhu, n_getaran, n_suara])

    return features_normalized


# ---- 1.Membaca Data Dari Text ---- #
data_features = []
data_targets = []
with open("fabric_machine.csv", "r") as file:
    lines = file.readlines()[1:]  # Skip header
    for line in lines:
        kolom = line.strip().split(",")
        suhu, getaran, suara, status = extract_data(kolom)
        print(f"cekkce {suhu} {getaran} {suara} {status}")
        data_features.append([suhu, getaran, suara])
        data_targets.append(status)

# ---- 2. Normalisasi Data (MIN - MAX SCALING) ---- #
suhu_all = [feature[0] for feature in data_features]
getaran_all = [feature[1] for feature in data_features]
suara_all = [feature[2] for feature in data_features]

suhu_minmax = min(suhu_all), max(suhu_all)
getaran_minmax = min(getaran_all), max(getaran_all)
suara_minmax = min(suara_all), max(suara_all)

print(f"Range Suhu: {suhu_minmax[0]} - {suhu_minmax[1]}")
print(f"Range Getaran: {getaran_minmax[0]} - {getaran_minmax[1]}")
print(f"Range Suara: {suara_minmax[0]} - {suara_minmax[1]}")

features_normalized = normalize_features(
    data_features, suhu_minmax, getaran_minmax, suara_minmax
)

X = torch.tensor(features_normalized, dtype=torch.float32)
y = torch.tensor(data_targets, dtype=torch.long)


# --- 3. BIKIN MODEL AI (MULTI-CLASS) ---
class FactoryMachineMonitoringAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(3, 8)
        self.output_layer = nn.Linear(8, 4)

    def forward(self, X):
        z1 = torch.relu(self.hidden_layer(X))
        return self.output_layer(z1)


model = FactoryMachineMonitoringAI()
# model = load_model(model, factory_machine_monitoring_brain_path)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


# ---- 4. TRAINING AI ---- #
# if not is_model_exists(factory_machine_monitoring_brain_path):
print(f"Training Model.....")
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    # print loss setiap 1000 epoch
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# ---- 5. UJIAN ---- #
new_machines = []
plain_new_machines = []
tensor_new_machines = None
with open("new_machine.csv", "r") as file:
    lines = file.readlines()[1:]  # Skip header
    for line in lines:
        kolom = line.strip().split(",")
        suhu, getaran, suara, _ = extract_data(kolom)
        new_suhu = calculate_min_max(suhu, suhu_minmax[0], suhu_minmax[1])
        new_getaran = calculate_min_max(getaran, getaran_minmax[0], getaran_minmax[1])
        new_suara = calculate_min_max(suara, suara_minmax[0], suara_minmax[1])
        new_machines.append([new_suhu, new_getaran, new_suara])
        plain_new_machines.append([suhu, getaran, suara])

tensor_new_machines = torch.tensor(new_machines, dtype=torch.float32)

status_dictionary = {
    0: "Normal 🟢",
    1: "Waspada 🟡 (Butuh perawatan minggu depan)",
    2: "Bahaya 🟠 (Harus dimatikan hari ini)",
    3: "Rusak Parah 🔴 (Sudah meledak/hancur)",
}

with torch.no_grad():
    prediction = model(tensor_new_machines)
    print(f"Prediction Result {prediction}")
    decision_threshold = torch.argmax(prediction, dim=1)
    print(f"decision threshold {decision_threshold}")

    for i in range(len(plain_new_machines)):
        suhu, getaran, suara, _ = extract_data(plain_new_machines[i])
        prediction_result = status_dictionary[(int(decision_threshold[i].item()))]
        print(
            f"Machine {i+1}: \nsuhu={suhu} getaran={getaran} suara={suara}\n status={prediction_result}"
        )

save_model(model, factory_machine_monitoring_brain_path)
