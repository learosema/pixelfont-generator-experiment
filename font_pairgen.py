from PIL import Image
import numpy as np
import random
import os

CHAR_W = CHAR_H = 8
GRID_W = GRID_H = 16
IMG_W = CHAR_W * GRID_W
IMG_H = CHAR_H * GRID_H

def mask_random_chars(base_img: Image.Image, keep_count: int) -> Image.Image:
    img_arr = np.array(base_img)
    masked_arr = np.zeros_like(img_arr)

    indices = list(range(256))
    keep = random.sample(indices, keep_count)

    for i in keep:
        row = i // 16
        col = i % 16
        x = col * CHAR_W
        y = row * CHAR_H
        masked_arr[y:y+CHAR_H, x:x+CHAR_W] = img_arr[y:y+CHAR_H, x:x+CHAR_W]

    return Image.fromarray(masked_arr, mode="L")

def make_training_pair(masked: Image.Image, full: Image.Image) -> Image.Image:
    pair = Image.new("L", (IMG_W * 2, IMG_H))
    pair.paste(masked, (0, 0))
    pair.paste(full, (IMG_W, 0))
    return pair.resize((512, 256), Image.NEAREST)

def process_all_fonts(input_dir: str, output_dir: str, num_variants: int):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".png"):
            continue

        font_path = os.path.join(input_dir, filename)
        font_name = os.path.splitext(filename)[0]
        print(f"Processing {filename}")

        try:
            base_img = Image.open(font_path).convert("L")

            if base_img.size != (128, 128):
                print(f"Skipping {filename}: not 128x128")
                continue

            for i in range(num_variants):
                count = random.randint(6, 24)
                masked = mask_random_chars(base_img, count)
                pair = make_training_pair(masked, base_img)
                out_path = os.path.join(output_dir, f"{font_name}_v{i}.png")
                pair.save(out_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


