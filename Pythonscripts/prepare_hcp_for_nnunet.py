import os
import re
import subprocess
import nibabel as nib
import numpy as np
from tqdm import tqdm
import json

# ==== Base paths ====
PARENT = "/home/m512f/dev/data/HCP"
NNUNET_RAW = "/home/m512f/dev/HCP-nnUnetSetup/nnunet_raw"
SCRIPT1 = os.path.join(os.path.dirname(__file__), "splitpeaks.py")
SCRIPT2 = os.path.join(os.path.dirname(__file__), "mergelabels.py")

# ==== Create DatasetXXX ====
existing = [d for d in os.listdir(NNUNET_RAW) if re.match(r"Dataset\d{3}.*", d)]
next_num = max([int(re.findall(r"\d{3}", d)[0]) for d in existing], default=0) + 1
dataset_name = f"Dataset{next_num:03d}_OpticRadiation"
dataset_dir = os.path.join(NNUNET_RAW, dataset_name)
imagesTr = os.path.join(dataset_dir, "imagesTr")
labelsTr = os.path.join(dataset_dir, "labelsTr")
imagesTs = os.path.join(dataset_dir, "imagesTs")
mapping_file = os.path.join(dataset_dir, "patient_id_mapping.txt")

os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)
os.makedirs(imagesTs, exist_ok=True)
print(f"📁 Creating dataset folder: {dataset_name}\n")

# ==== Step 1: Split 4D peaks ====
print("STEP 1: Splitting 4D peaks...")
subprocess.run([
    "python3", SCRIPT1,
    "--parent_folder", PARENT,
    "--output_folder", imagesTr,
    "--mapping_file", mapping_file
], check=True)

# ==== Step 2: Merge OR labels ====
print("\nSTEP 2: Merging OR labels...")
result = subprocess.run([
    "python3", SCRIPT2,
    "--parent_folder", PARENT,
    "--output_folder", labelsTr,
    "--mapping_file", mapping_file
], capture_output=True, text=True, check=True)
print(result.stdout)

# ==== Step 3: Filter images to only include patients with labels ====
label_patient_ids = [f.split(".")[0] for f in os.listdir(labelsTr) if f.endswith(".nii.gz")]

for img_file in os.listdir(imagesTr):
    pid = img_file.split("_")[0]  # filenames like 001_0000.nii.gz
    if pid not in label_patient_ids:
        os.remove(os.path.join(imagesTr, img_file))

print(f"\n✅ Filtered imagesTr to only include patients with labels ({len(label_patient_ids)} patients)")

# ==== Step 4: Generate dataset.json with proper channels ====
print("\nSTEP 3: Generating dataset.json (nnU-Net format)...")

training = []
max_channels = 0
for pid in label_patient_ids:
    image_channels = sorted([os.path.join("imagesTr", f)
                             for f in os.listdir(imagesTr) if f.startswith(pid)])
    training.append({"image": image_channels, "label": os.path.join("labelsTr", f"{pid}.nii.gz")})
    max_channels = max(max_channels, len(image_channels))

# Create channel names dynamically: peak0, peak1, ..., peakN
channel_names = {i: f"peak{i}" for i in range(max_channels)}

dataset_json = {
    "channel_names": channel_names,
    "labels": {"background": 0, "left_or": 1, "right_or": 2},
    "numTraining": len(training),
    "file_ending": ".nii.gz",
    "name": dataset_name,
    "description": "Optic radiation segmentation dataset (HCP)",
    "reference": "Human Connectome Project",
    "licence": "For research use only",
    "release": "1.0",
    "training": training,
    "test": []
}

with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ dataset.json written to {os.path.join(dataset_dir, 'dataset.json')}")
print("\n🎉 DONE!")
print(f"📦 Dataset folder: {dataset_dir}")
print(f"🗂️ Mapping file: {mapping_file}")
