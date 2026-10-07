import os
import torch


def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print("Model loaded from", path)
    else:
        print("Model not found at", path, "- starting from scratch.")
    return model


def save_model(model, path):
    # Buat folder jika belum ada
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.save(model.state_dict(), path)
    print("Model saved to", path)
