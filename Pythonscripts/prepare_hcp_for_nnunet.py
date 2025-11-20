# prepare_hcp_for_nnunet.py

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

# ==== Define test fold (fold5) ====
TEST_FOLD = {
    "fold5": ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']
}
TEST_SUBJECTS = set(TEST_FOLD["fold5"])

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
    "--output_folder", imagesTr,  # Initially put all in imagesTr
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

# ==== Step 3: Load mapping to identify test subjects ====
print("\nSTEP 3: Separating test subjects (fold5) into imagesTs...")
mapping = {}
if os.path.exists(mapping_file):
    with open(mapping_file) as f:
        for line in f:
            if "->" in line:
                num, orig = line.strip().split("->")
                mapping[orig.strip()] = num.strip()

# Identify which new IDs correspond to test subjects
test_new_ids = set()
for orig_id, new_id in mapping.items():
    if orig_id in TEST_SUBJECTS:
        test_new_ids.add(new_id)

print(f"🔍 Identified {len(test_new_ids)} test subjects from fold5")

# ==== Step 4: Move test subjects from imagesTr to imagesTs ====
moved_count = 0
for img_file in os.listdir(imagesTr):
    pid = img_file.split("_")[0]  # filenames like 001_0000.nii.gz
    if pid in test_new_ids:
        # Move from imagesTr to imagesTs
        src_path = os.path.join(imagesTr, img_file)
        dst_path = os.path.join(imagesTs, img_file)
        os.rename(src_path, dst_path)
        moved_count += 1

print(f"📁 Moved {moved_count} image files from imagesTr to imagesTs")

# ==== Step 5: Create labelsTs folder for test labels ====
labelsTs = os.path.join(dataset_dir, "labelsTs")
os.makedirs(labelsTs, exist_ok=True)

# Move test labels to labelsTs instead of deleting
moved_label_count = 0
for label_file in os.listdir(labelsTr):
    pid = label_file.split(".")[0]
    if pid in test_new_ids:
        src_path = os.path.join(labelsTr, label_file)
        dst_path = os.path.join(labelsTs, label_file)
        os.rename(src_path, dst_path)
        moved_label_count += 1

print(f"📁 Moved {moved_label_count} label files from labelsTr to labelsTs")

# ==== Step 6: Filter remaining imagesTr to only include patients with labels ====
label_patient_ids = [f.split(".")[0] for f in os.listdir(labelsTr) if f.endswith(".nii.gz")]

final_removed_count = 0
for img_file in os.listdir(imagesTr):
    pid = img_file.split("_")[0]
    if pid not in label_patient_ids:
        os.remove(os.path.join(imagesTr, img_file))
        final_removed_count += 1

print(f"🔧 Filtered imagesTr: removed {final_removed_count} additional files without labels")
print(f"📊 Final counts: {len(label_patient_ids)} training subjects, {len(test_new_ids)} test subjects")
print(f"✅ Test labels preserved in: {labelsTs}")

# ==== Step 7: Generate dataset.json with proper channels ====
print("\nSTEP 4: Generating dataset.json (nnU-Net format)...")

# Training data - only subjects that have both images and labels
training = []
max_channels = 0
for pid in label_patient_ids:
    image_channels = sorted([os.path.join("imagesTr", f)
                             for f in os.listdir(imagesTr) if f.startswith(pid)])
    training.append({"image": image_channels, "label": os.path.join("labelsTr", f"{pid}.nii.gz")})
    max_channels = max(max_channels, len(image_channels))

# Test data - subjects in imagesTs
test_images = []
for pid in test_new_ids:
    image_channels = sorted([os.path.join("imagesTs", f)
                            for f in os.listdir(imagesTs) if f.startswith(pid)])
    test_images.append(image_channels)

# Create channel names dynamically: peak0, peak1, ..., peakN
channel_names = {i: f"peak{i}" for i in range(max_channels)}

dataset_json = {
    "channel_names": channel_names,
    "labels": {"background": 0, "left_or": 1, "right_or": 2},
    "numTraining": len(training),
    "file_ending": ".nii.gz",
    "name": dataset_name,
    "description": "Optic radiation segmentation dataset (HCP) - fold5 as test set",
    "reference": "Human Connectome Project",
    "licence": "For research use only",
    "release": "1.0",
    "training": training,
    "test": test_images,
    "test_labels_available": True,
    "test_labels_path": "labelsTs"  # Relative to dataset directory
}

with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ dataset.json written to {os.path.join(dataset_dir, 'dataset.json')}")
print("\n🎉 DONE!")
print(f"📦 Dataset folder: {dataset_dir}")
print(f"🗂️ Mapping file: {mapping_file}")
print(f"📊 Training subjects: {len(training)}, Test subjects: {len(test_images)}")