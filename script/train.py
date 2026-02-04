# script/train.py

import torch
import os
from torch.utils.data import DataLoader
from dataset import SegDataset
from model import get_model
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
from tqdm import tqdm

# ==============================
# CONFIG (PRO SETTINGS)
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 14
LR = 3e-4                 # cosine works better with slightly higher LR
EPOCHS = 50
NUM_CLASSES = 10

TRAIN_IMG = "dataset/train/images"
TRAIN_MASK = "dataset/train/masks"
VAL_IMG = "dataset/val/images"
VAL_MASK = "dataset/val/masks"

SAVE_PATH = "checkpoints/best.pth"

# ==============================
# DATA
# ==============================

train_ds = SegDataset(TRAIN_IMG, TRAIN_MASK, train=True)
val_ds   = SegDataset(VAL_IMG, VAL_MASK, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================
# MODEL
# ==============================

model = get_model().to(DEVICE)

if os.path.exists(SAVE_PATH):
    model.load_state_dict(torch.load(SAVE_PATH))
    print("✅ Loaded checkpoint — resuming")

print("🔥 Using device:", DEVICE)

# ==============================
# LOSS (STRONG COMBO)
# ==============================

dice_loss  = smp.losses.DiceLoss("multiclass")
focal_loss = smp.losses.FocalLoss("multiclass")
ce_loss    = torch.nn.CrossEntropyLoss()

def loss_fn(preds, masks):
    return (
        0.4 * ce_loss(preds, masks) +
        0.3 * dice_loss(preds, masks) +
        0.3 * focal_loss(preds, masks)
    )

# ==============================
# OPTIMIZER
# ==============================

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# 🔥 Cosine schedule = smoother training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

scaler = torch.cuda.amp.GradScaler()

# ==============================
# TRAIN LOOP
# ==============================

best_iou = 0
patience = 8
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, masks in loop:

        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss = loss_fn(preds, masks)

        scaler.scale(loss).backward()

        # 🔥 gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    train_loss /= len(train_loader)

    # ================= VALIDATION =================

    model.eval()
    val_iou = 0

    with torch.no_grad():
        for imgs, masks in val_loader:

            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            preds = torch.argmax(preds, dim=1)

            val_iou += metric(preds, masks).item()

    val_iou /= len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val IoU: {val_iou:.4f}")

    # ================= SAVE BEST =================

    if val_iou > best_iou:
        best_iou = val_iou
        patience_counter = 0
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Saved Best Model")

    else:
        patience_counter += 1

    # ================= EARLY STOP =================

    if patience_counter >= patience:
        print("🛑 Early stopping")
        break


print("\n🎯 Training complete")
print("Best IoU:", best_iou)
