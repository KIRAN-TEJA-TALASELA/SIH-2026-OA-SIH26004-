import os
import csv
import random 
import hashlib 
from pathlib import Path 
from collections import Counter 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image, ImageStat 

DATA_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\DATA")
RESULTS_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\RESULTS")

SPLITS=["train","val","test"]
CLASSES=["0","1","2","3","4"]
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

random.seed(42)

print("\n==============================================")
print("       OsteoSense Dataset Visualization")
print("==============================================")
def get_image_files(folder):
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def calculate_md5(path):
    md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda:f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None

def inspect_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size 
            mode = img.mode 
            image_array = np.array(img.convert("L"), dtype = np.float32)

            mean_intensity = float(np.mean(image_array))
            std_intensity = float(np.std(image_array))
            min_intensity = float(np.min(image_array))
            max_intensity = float(np.max(image_array))

            return {
                "valid" : True,
                "width" :width,
                "height" : height,
                "aspect_ratio" : width/height if height!=0 else 0,
                "mode" : mode,
                "mean_intensity" : mean_intensity,
                "std_intensity" : std_intensity,
                "min_intensity" : min_intensity,
                "max_intensity" : max_intensity
            }
    except Exception as e:
        return {
            "valid" : False,
            "error" : str(e)
        }

all_records = []
corrupted_files = []
duplicate_hashes = {}

print("\nScanning Datasets............")

for split in SPLITS:
    split_dir = DATA_DIR/split
    print(f"\n----{split.upper()}----")
    if not split_dir.exists():
        print(f"WARNING: {split_dir} does not exist")
        continue
    for class_name in CLASSES:
        class_dir = split_dir/class_name

        if not class_dir.exists():
            print(f"KL {class_name}: folder not found")
            continue

        files = get_image_files(class_dir)
        print(f"KL {class_name}: {len(files)} images")

        for path in files:
            info = inspect_image(path)
            record = {
                "split" : split,
                "class" : class_name,
                "filename" : path.name,
                "path" : str(path)
            }

            record.update(info)
            all_records.append(record)
            if not info["valid"]:
                corrupted_files.append(record)
                continue
            file_hash = calculate_md5(path)
            if file_hash is not None:
                if file_hash not in duplicate_hashes:
                    duplicate_hashes[file_hash] = []
                duplicate_hashes[file_hash].append(str(path))

print("\n==============================================")
print("Dataset Scan Complete")
print("==============================================")

total_images=len(all_records)
valid_images=sum(1 for r in all_records if r["valid"])
corrupted_count=len(corrupted_files)

print(f"Total images: {total_images}")
print(f"Valid images: {valid_images}")
print(f"Corrupted images: {corrupted_count}")

duplicate_groups = {h:paths
                    for h, paths in duplicate_hashes.items() if len(paths) > 1}

duplicate_image_count = sum(len(paths) for paths in duplicate_groups.values())
print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Images involved in duplicates: {duplicate_image_count}")

csv_path=RESULTS_DIR/"dataset_image_statistics.csv"

fieldnames = ["split", "class", "filename", "path", "valid", "width", "height", "aspect_ratio", "mode", "mean_intensity", "std_intensity", "min_intensity", "max_intensity"]

with open(csv_path, "w", newline="", encoding= "utf-8") as f:
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()

    for record in all_records:
        writer.writerow({key:record.get(key, "") for key in fieldnames})

print(f"\nSaved statistics:")
print(csv_path)

corrupt_path=RESULTS_DIR/"corrupted_images.txt"

with open(corrupt_path, "w", encoding= "utf-8") as f:
    if corrupted_files:
        f.write(
            f"{record['split']} |"
            f"KL {record['class']} |"
            f"{record['path']} |"
            f"{record.get('error', '')}\n"
        )
    else:
        f.write("No corrupt images detected")

duplicate_path=RESULTS_DIR/"duplicate_images.txt"

with open(duplicate_path, "w", encoding= "utf-8") as f:
    if duplicate_groups:
        for index, (file_hash, paths) in enumerate(duplicate_groups.items, start= 1):
            f.write(f"\nDuplicate Group {index}")
            f.write(f"MD5: {file_hash}\n")

            for path in paths:
                f.write(f"{path}\n")
    else:
        f.write("No exact duplicate images detected\n")

print("\n==============================================")
print("Image Property Analysis")
print("==============================================")

valid_records = [r for r in all_records if r['valid']]

widths = [r['width'] for r in valid_records]
heights = [r['height'] for r in valid_records]
aspect_ratios = [r['aspect_ratio'] for r in valid_records]
mean_intensities = [r['mean_intensity'] for r in valid_records]
std_intensities = [r['std_intensity'] for r in valid_records]

if valid_records:
    print(
        f"\nWidth: "
        f"min = (min{widths})"
        f"max = max(max{widths})"
        f"mean = {np.mean(widths):.1f}"
    )

    print(
        f"Height: "
        f"min={min(heights)}, "
        f"max={max(heights)}, "
        f"mean={np.mean(heights):.1f}"
    )

    print(
        f"Aspect ratio: "
        f"min={min(aspect_ratios):.3f}, "
        f"max={max(aspect_ratios):.3f}, "
        f"mean={np.mean(aspect_ratios):.3f}"
    )

    print(
        f"Mean intensity: "
        f"min={min(mean_intensities):.2f}, "
        f"max={max(mean_intensities):.2f}, "
        f"mean={np.mean(mean_intensities):.2f}"
    )

    print(
        f"Intensity SD: "
        f"mean={np.mean(std_intensities):.2f}"
    )

mode_counts = Counter(r['mode'] for r in valid_records)
print("\nImages nodes: ")
for mode, count in mode_counts.items():
    print(f"{mode} : {count}")

print("\n==============================================")
print("Class Distribution")
print("==============================================")

distribution=[]

for split in SPLITS:
    print(f"\n{split.upper()}")
    split_records = [r for r in valid_records if r['split'] == split]

    for class_name in CLASSES:
        count = sum(1 for r in split_records if r['class'] == class_name)
        distribution.append((split, class_name, count))
        print(f"KL {class_name}: {count}")

plt.figure(figsize=(10,6))

x=np.arange(len(CLASSES))
width=0.25

for i, split in enumerate(SPLITS):
    counts = [count for s, c, counts in distribution if s == split]
    plt.bar(x + (i-1)*width, counts, width, label = split)
plt.xticks(x, ['KL 0', 'KL 1', 'KL 2', 'KL 3', 'KL 4'])
plt.xlabel("Kellgren-Lawrence Grade")
plt.ylabel("Number of Images")
plt.title("OsteoSense Dataset Class Distribution")
plt.legend()
plt.tight_layout()
distribution_plot=RESULTS_DIR/"class_distribution.png"
plt.savefig(distribution_plot, dpi=300)
plt.close()
print(f"\nSaved: {distribution_plot}")
print("\n==============================================")
print("Representative Image Montage")
print("==============================================")
fig, axes = plt.subplots(5, 5, figsize = (15, 15))

for row, class_name in enumerate(CLASSES):
    class_records = [r for r in valid_records if r['split'] == 'train' and r['class'] == class_name]
    if not class_records:
        for col in range(5):
            axes[row, col].axis('off')
        continue
    sample_count = min(5, len(class_records))
    samples = random.sample(class_records, sample_count)
    for col, sample in enumerate(samples):
        image_path = Path(sample['path'])
        try:
            image = Image.open(image_path).convert('L')
            axes[row, col].imshow(image, cmap = 'gray')
            axes[row, col].set_title(f"KL {class_name}\n" f"{image.size[0]}x{image.size[1]}")
            axes[row, col].axis('off')
        except Exception:
            axes[row, col].axis('off')
fig.suptitle("OsteoSense - Representative Training X-rays", fontsize = 18)
plt.tight_layout()
montage_path=RESULTS_DIR/"training_xray_montage.png"
plt.savefig(montage_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {montage_path}")

print("\n==============================================")
print("Aspect Ratio Distribution")
print("==============================================")
plt.figure(figsize=(9,6))
plt.hist(aspect_ratios, bins=30)
plt.xlabel("Width / Height")
plt.ylabel("Number of Images")
plt.title("OsteoSense - Image Aspect Ratio Distribution")
plt.tight_layout()
aspect_path=RESULTS_DIR/"aspect_ratio_distribution.png"
plt.savefig(aspect_path, dpi=300)
plt.close()
print(f"Saved: {aspect_path}")

print("\n==============================================")
print("Intensity Distribution")
print("==============================================")
plt.figure(figsize=(9,6))
plt.hist(mean_intensities, bins=30)
plt.xlabel("Mean Grayscale Intensity")
plt.ylabel("Number of Images")
plt.title("OsteoSense - Mean Image Intensity Distribution")
plt.tight_layout()
intensity_path=RESULTS_DIR/"intensity_distribution.png"
plt.savefig(intensity_path, dpi=300)

plt.close()

print(f"Saved: {intensity_path}")

print("\n==============================================")
print("Dataset Visualization Complete")
print("==============================================")

print("\nFiles generated in RESULTS:")

print("dataset_image_statistics.csv")
print("corrupted_images.txt")
print("duplicate_images.txt")
print("class_distribution.png")
print("training_xray_montage.png")
print("aspect_ratio_distribution.png")
print("intensity_distribution.png")

print("\n")