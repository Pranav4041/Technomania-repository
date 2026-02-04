# script/test.py

import torch
from torch.utils.data import DataLoader
from dataset import SegDataset
from model import get_model
from torchmetrics import JaccardIndex
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import shutil


# ==============================
# CONFIG
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
BATCH_SIZE = 4

TEST_IMG = "testing/images"
TEST_MASK = "testing/masks"
MODEL_PATH = "checkpoints/best.pth"

FAIL_THRESHOLD = 0.20
FAIL_DIR = "failures"

os.makedirs(f"{FAIL_DIR}/images", exist_ok=True)
os.makedirs(f"{FAIL_DIR}/masks", exist_ok=True)


# ==============================
# SAFE FILE ORDER (important)
# ==============================

img_files = sorted(os.listdir(TEST_IMG))
mask_files = sorted(os.listdir(TEST_MASK))


# ==============================
# DATA
# ==============================

test_ds = SegDataset(TEST_IMG, TEST_MASK, train=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"🔥 Testing on {len(test_ds)} images")


# ==============================
# MODEL
# ==============================

model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")


# ==============================
# METRIC
# ==============================

metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)


# ==============================
# TEST LOOP
# ==============================

total_iou = 0
all_preds = []
all_labels = []

failure_preds = []
failure_labels = []
failure_ious = []   # ⭐ NEW (store failure IoUs)

per_image_ious = []

idx = 0

with torch.no_grad():
    for imgs, masks in tqdm(test_loader):

        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        for p, m in zip(preds, masks):

            iou = metric(p.unsqueeze(0), m.unsqueeze(0)).item()

            per_image_ious.append(iou)
            total_iou += iou

            all_preds.append(p.cpu().numpy())
            all_labels.append(m.cpu().numpy())

            img_name = img_files[idx]
            mask_name = mask_files[idx]

            # ======================
            # FAILURE CASE
            # ======================
            if iou < FAIL_THRESHOLD:

                print(f"❌ {img_name} | IoU = {iou:.4f}")

                failure_ious.append(iou)   # ⭐ NEW

                failure_preds.append(p.cpu().numpy())
                failure_labels.append(m.cpu().numpy())

                shutil.copy(
                    f"{TEST_IMG}/{img_name}",
                    f"{FAIL_DIR}/images/{img_name}"
                )

                shutil.copy(
                    f"{TEST_MASK}/{mask_name}",
                    f"{FAIL_DIR}/masks/{mask_name}"
                )

            idx += 1


# ==============================
# METRICS
# ==============================

mean_iou = total_iou / len(per_image_ious)

print("\n🎯 FULL TEST RESULTS")
print("======================")
print(f"Mean IoU : {mean_iou:.4f}")
print(f"Failures (<{FAIL_THRESHOLD}) : {len(failure_preds)}")
print("======================")


# ==============================
# CONFUSION MATRIX FUNCTION
# ==============================

def plot_cm(preds, labels, title, filename):

    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    cm = confusion_matrix(labels, preds)
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{filename}")
    plt.close()


# ==============================
# CONFUSION MATRICES
# ==============================

print("📊 Generating confusion matrices...")

plot_cm(all_preds, all_labels, "Full Test Confusion Matrix", "confusion_full.png")

if len(failure_preds) > 0:
    plot_cm(failure_preds, failure_labels,
            "Failure Only Confusion Matrix", "confusion_failures.png")


# ==============================
# ⭐ FAILURE IoU GRAPH (NEW)
# ==============================

if len(failure_ious) > 0:

    plt.figure(figsize=(10,5))
    plt.bar(range(len(failure_ious)), failure_ious)
    plt.axhline(y=FAIL_THRESHOLD, linestyle="--")

    plt.xlabel("Failure Case Index")
    plt.ylabel("IoU")
    plt.title("IoU Scores of Failure Cases")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/failure_ious.png")
    plt.close()

    print("📉 Saved → outputs/failure_ious.png")


print("✅ Done")
