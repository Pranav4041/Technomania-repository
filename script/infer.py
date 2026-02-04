# script/infer.py

import os
import cv2
import torch
import numpy as np
from model import get_model
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10

IMG_DIR = "testing/images"
MODEL_PATH = "checkpoints/best.pth"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================

model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("🔥 Model loaded for inference")

# ==============================
# RANDOM COLORS
# ==============================

colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3))

def colorize(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(NUM_CLASSES):
        out[mask == c] = colors[c]

    return out

# ==============================
# INFERENCE LOOP
# ==============================

files = os.listdir(IMG_DIR)

for name in tqdm(files):

    path = os.path.join(IMG_DIR, name)

    img = cv2.imread(path)
    if img is None:
        continue

    original = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    mask_color = colorize(pred)
    overlay = cv2.addWeighted(original, 0.6, mask_color, 0.4, 0)

    base = os.path.splitext(name)[0]

    cv2.imwrite(f"{OUT_DIR}/{base}_mask.png", mask_color)
    cv2.imwrite(f"{OUT_DIR}/{base}_overlay.png", overlay)

print("\n✅ Outputs saved in /outputs/")
