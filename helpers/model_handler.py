import os
import torch


def is_model_exists(path):
    return os.path.exists(path)


def load_model(model, path):
    if not is_model_exists(path):
        print("Model not found at", path, "- starting from scratch.")
        return model

    model.load_state_dict(torch.load(path))
    print("Model loaded from", path)
    return model


def save_model(model, path):
    # Buat folder jika belum ada
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.save(model.state_dict(), path)
    print("Model saved to", path)
