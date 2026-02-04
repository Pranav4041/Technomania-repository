# 🔥 Semantic Segmentation Pipeline (PyTorch)

A complete deep learning pipeline for **multi-class semantic image segmentation** built using **PyTorch + segmentation_models_pytorch**.

This project includes:

✅ Training  
✅ Validation  
✅ IoU evaluation  
✅ Failure case detection  
✅ Confusion matrices  
✅ Failure IoU visualization  

Built for fast experimentation, debugging, and model improvement.

---

# 🚀 Features

### 🧠 Training
- UNet/DeepLab/any SMP model (from `segmentation_models_pytorch`)
- Dice + Focal + CrossEntropy hybrid loss
- AdamW optimizer
- Cosine LR scheduler
- Gradient clipping
- Automatic best model saving

### 📊 Evaluation
- Mean IoU (Jaccard Index)
- Per-image IoU computation
- Confusion matrix (full dataset)
- Confusion matrix (failures only)

### ❌ Failure Analysis (Debug Mode)
Automatically:
- detects images with IoU < threshold
- saves failed images + masks
- prints IoU scores
- plots IoU graph for failure distribution

This helps quickly answer:
> Where is the model failing and how badly?

---

# 📂 Project Structure

project/
│
├── dataset/
│ ├── train/images
│ ├── train/masks
│ ├── val/images
│ ├── val/masks
│
├── testing/
│ ├── images
│ ├── masks
│
├── script/
│ ├── train.py
│ ├── test.py
│ ├── dataset.py
│ ├── model.py
│
├── checkpoints/
│ └── best.pth
│
├── failures/
│ ├── images/
│ ├── masks/
│
├── outputs/
│ ├── confusion_full.png
│ ├── confusion_failures.png
│ ├── failure_ious.png
│
└── README.md


---

# ⚙️ Installation

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install torchmetrics
pip install matplotlib seaborn tqdm

🏋️ Training
python script/train.py

Settings

Epochs: 30

Batch size: 14

Loss: CE + Dice + Focal

Workers: 0 (Windows-safe)

Best model saved automatically

Output:

checkpoints/best.pth

🧪 Testing
python script/test.py

Outputs

Mean IoU score

Failure cases printed with IoU

Failed images copied to:

failures/


Graph of failure IoUs:

outputs/failure_ious.png


Confusion matrices:

outputs/confusion_full.png
outputs/confusion_failures.png

📉 Failure IoU Graph

Each bar = one failed image
Helps identify:

near-threshold misses (minor errors)

catastrophic failures

data quality issues

🧠 Tech Stack

PyTorch

segmentation_models_pytorch

TorchMetrics

Matplotlib

Seaborn

💡 Why this project?

Most segmentation repos only show:

"accuracy = 0.9"

Which is useless.

This pipeline focuses on:

per-image analysis

failure debugging

real-world reliability

Because models don’t fail on averages —
they fail on specific samples.

🔮 Possible Improvements

Overlay prediction vs ground truth

Per-class IoU

Mixed precision training

Data augmentation

Faster dataloaders

WandB / TensorBoard logging

👤 Author

Pranav
Dhruv
Mudit
AI/ML Engineering Student
Built for practical experimentation and model debugging.

🧊 Brutal Truth

If failure IoUs are very low (< 0.1):
→ model isn’t "slightly wrong"
→ it’s guessing

Fix:

labels

class imbalance

augmentation

loss weights

Not evaluation code.


---

If you want, next we can upgrade README with:
- badges
- screenshots of confusion matrix
- sample predictions
- demo GIF
- or make it resume-ready for internships/portfolio

Just say the vibe you want (minimal / pro / flashy).
