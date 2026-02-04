import os
import cv2
import random
import shutil
import numpy as np
import albumentations as A


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FAIL_IMG_DIR = "failures/images"
FAIL_MASK_DIR = "failures/masks"

OUT_IMG_DIR = "augmented/images"
OUT_MASK_DIR = "augmented/masks"

NUM_AUGS = 20   # 🔥 24 × 20 = 480 images



if os.path.exists("augmented"):
    shutil.rmtree("augmented")

os.makedirs(OUT_IMG_DIR)
os.makedirs(OUT_MASK_DIR)



augment = A.Compose([
    A.RandomBrightnessContrast(p=0.4),
    A.MotionBlur(blur_limit=5, p=0.25),
    A.GaussNoise(p=0.25),
    A.RandomShadow(p=0.2),
    A.RandomFog(p=0.15),

    A.RandomRain(
        slant_range=(-5, 5),
        drop_length=6,
        drop_width=1,
        blur_value=3,
        brightness_coefficient=0.9,
        p=0.15
    ),

    A.RandomSunFlare(p=0.1),
])



def add_sandstorm(img):
    dust = np.random.normal(180, 20, img.shape).astype(np.uint8)
    img = cv2.addWeighted(img, 0.9, dust, 0.1, 0)
    return cv2.GaussianBlur(img, (3, 3), 0)


def add_sunset(img):
    overlay = np.full_like(img, (0, 90, 200))
    return cv2.addWeighted(img, 0.85, overlay, 0.15, 0)


def add_night(img):
    gamma = 1.3
    return ((img / 255.0) ** gamma * 255).astype(np.uint8)


def add_fog_soft(img):
    fog = np.random.uniform(200, 255, img.shape).astype(np.uint8)
    return cv2.addWeighted(img, 0.9, fog, 0.1, 0)


effects = [
    add_sandstorm,
    add_sunset,
    add_night,
    add_fog_soft
]



def augment_pair(img_path, mask_path, base_name):

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        return

    for i in range(NUM_AUGS):

        aug = augment(image=img, mask=mask)

        aug_img = aug["image"]
        aug_mask = aug["mask"]

        # apply exactly ONE subtle effect
        effect = random.choice(effects)
        aug_img = effect(aug_img)

        aug_mask = aug_mask.astype(np.uint8)

        cv2.imwrite(f"{OUT_IMG_DIR}/{base_name}_aug{i}.png", aug_img)
        cv2.imwrite(f"{OUT_MASK_DIR}/{base_name}_aug{i}.png", aug_mask)



def main():

    files = sorted(os.listdir(FAIL_IMG_DIR))

    if not files:
        print("❌ No failure images found")
        return

    total = len(files) * NUM_AUGS

    print(f"🚀 Found {len(files)} failure images")
    print(f"Generating {NUM_AUGS} per image")
    print(f"👉 Total output will be: {total} images")

    for file in files:
        augment_pair(
            os.path.join(FAIL_IMG_DIR, file),
            os.path.join(FAIL_MASK_DIR, file),
            os.path.splitext(file)[0]
        )

    print("\n✅ Augmentation complete!")
    print(f"📦 {total} images saved inside augmented/")



if __name__ == "__main__":
    main()
