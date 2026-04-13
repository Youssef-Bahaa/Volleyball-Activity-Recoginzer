import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from src.dataset.utils import  id2activity
from src.dataset.DataLoader.B1_loader import VolleyBallFeaturesDataset
from src.models.B1.B1_model import ResNetFineTune


# ─────────────────────────── Config ────────────────────────
IMAGE_ROOT = "data/videos"
ANNOT_PATH = "src/dataset/annot_all.pkl"
CHECKPOINT = "checkpoints/best_model.pth"
SAVE_DIR = "test_results"
NUM_CLASSES = 8
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ─────────────────────────── Transform ────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ─────────────────────────── Loader ───────────────────────────

def build_test_loader():
    dataset = VolleyBallFeaturesDataset(IMAGE_ROOT, ANNOT_PATH, test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return loader




# ─────────────────────────── Evaluation ───────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


# ─────────────────────────── Confusion Matrix ─────────────────

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels,preds)


# ─────────────────────────── Main ─────────────────
def test():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = ResNetFineTune(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = model.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Loaded checkpoint: {CHECKPOINT}")
    if "epoch" in ckpt:
        print(f"  trained for {ckpt['epoch']} epochs  |  saved val_acc: {ckpt.get('val_acc', 'n/a'):.4f}")

    print(f"Device: {DEVICE}\n")

    test_loader = build_test_loader()
    criterion   = nn.CrossEntropyLoss()

    loss, acc, labels, preds = evaluate(model, test_loader, criterion)

    report_path = os.path.join(SAVE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Loss     : {loss:.4f}\n")
        f.write(f"Test Accuracy : {acc:.4f}\n\n")
        f.write(classification_report(labels, preds, target_names=CLASS_NAMES, digits=3))
    print(f"  → Classification report saved to {report_path}")

    # ── Plots ──
    plot_confusion_matrix(labels, preds,os.path.join(SAVE_DIR, "confusion_matrix.png"))

    print("\nTesting complete.")


if __name__ == "__main__":
    test()


