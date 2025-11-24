# prepare_hcp_fa_for_nnunet.py

import os
import re
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm
import json

# ==== Base paths ====
PARENT = "/home/m512f/dev/data/HCP"
NNUNET_RAW = "/home/m512f/dev/HCP-nnUnetSetup/nnunet_raw"
SCRIPT2 = os.path.join(os.path.dirname(__file__), "mergelabels.py")

# ==== Define test fold (fold5) ====
TEST_FOLD = {
    "fold5": ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']
}
TEST_SUBJECTS = set(TEST_FOLD["fold5"])

# ==== Create Dataset002 ====
existing = [d for d in os.listdir(NNUNET_RAW) if re.match(r"Dataset\d{3}.*", d)]
next_num = 2  # Force Dataset002
dataset_name = f"Dataset{next_num:03d}_OpticRadiation"
dataset_dir = os.path.join(NNUNET_RAW, dataset_name)
imagesTr = os.path.join(dataset_dir, "imagesTr")
labelsTr = os.path.join(dataset_dir, "labelsTr")
imagesTs = os.path.join(dataset_dir, "imagesTs")
labelsTs = os.path.join(dataset_dir, "labelsTs")

os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)
os.makedirs(imagesTs, exist_ok=True)
os.makedirs(labelsTs, exist_ok=True)

print(f"📁 Creating dataset folder: {dataset_name}\n")

# ==== Step 1: Copy FA.nii.gz files ====
print("STEP 1: Copying FA.nii.gz files...")

# First, create a mapping file similar to the peaks version
mapping_file = os.path.join(dataset_dir, "patient_id_mapping.txt")
mapping_lines = []

patient_folders = sorted([
    os.path.join(PARENT, f)
    for f in os.listdir(PARENT)
    if os.path.isdir(os.path.join(PARENT, f))
])

# Copy FA files and create mapping
for i, patient_path in enumerate(tqdm(patient_folders, desc="Copying FA files", unit="patient"), start=1):
    patient_id = os.path.basename(patient_path)
    fa_path = os.path.join(patient_path, 'FA.nii.gz')
    
    if os.path.isfile(fa_path):
        # Copy FA file to imagesTr (we'll separate test/train later)
        new_filename = f"{i:03d}_0000.nii.gz"  # Single channel, always 0000
        dest_path = os.path.join(imagesTr, new_filename)
        shutil.copy2(fa_path, dest_path)
        
        mapping_lines.append(f"{i:03d} -> {patient_id}\n")
    else:
        print(f"⚠️ No FA.nii.gz found in {patient_id}")

# Save mapping file
with open(mapping_file, "w") as f:
    f.writelines(mapping_lines)

print(f"✅ Copied {len(mapping_lines)} FA files to imagesTr")

# ==== Step 2: Merge OR labels ====
print("\nSTEP 2: Merging OR labels...")
import subprocess
result = subprocess.run([
    "python3", SCRIPT2,
    "--parent_folder", PARENT,
    "--output_folder", labelsTr,
    "--mapping_file", mapping_file
], capture_output=True, text=True, check=True)
print(result.stdout)

# ==== Step 3: Load mapping to identify test subjects ====
print("\nSTEP 3: Separating test subjects (fold5)...")
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

# ==== Step 5: Move test labels to labelsTs ====
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

# ==== Step 7: Generate dataset.json for FA data ====
print("\nSTEP 7: Generating dataset.json (nnU-Net format)...")

# Training data - only subjects that have both images and labels
training = []
for pid in label_patient_ids:
    # FA data has only one channel: peak0
    image_path = os.path.join("imagesTr", f"{pid}_0000.nii.gz")
    training.append({"image": image_path, "label": os.path.join("labelsTr", f"{pid}.nii.gz")})

# Test data - subjects in imagesTs
test_images = []
for pid in test_new_ids:
    image_path = os.path.join("imagesTs", f"{pid}_0000.nii.gz")
    test_images.append(image_path)

dataset_json = {
    "channel_names": {"0": "FA"},  # Single channel: Fractional Anisotropy
    "labels": {"background": 0, "left_or": 1, "right_or": 2},
    "numTraining": len(training),
    "file_ending": ".nii.gz",
    "name": dataset_name,
    "description": "Optic radiation segmentation dataset (HCP) - FA only - fold5 as test set",
    "reference": "Human Connectome Project",
    "licence": "For research use only",
    "release": "1.0",
    "training": training,
    "test": test_images,
    "test_labels_available": True,
    "test_labels_path": "labelsTs"
}

with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ dataset.json written to {os.path.join(dataset_dir, 'dataset.json')}")

# ==== Step 8: Create splits_final.json using the same folds ====
print("\nSTEP 8: Creating splits_final.json...")

# Define the same folds as before (only folds 1-4 for training)
folds = {
    "fold1": ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574', '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241', '907656', '904044'],
    "fold2": ['901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579', '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', '859671', '857263', '856766'],
    "fold3": ['849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653', '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370', '771354', '770352', '765056'],
    "fold4": ['761957', '759869', '756055', '753251', '751348', '749361', '748662', '748258', '742549', '734045', '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133', '695768', '690152']
}

# Convert folds to new IDs
converted_folds = {}
for fold_name, subjects in folds.items():
    converted_subjects = [mapping[s] for s in subjects if s in mapping]
    converted_folds[fold_name] = converted_subjects
    print(f"📊 {fold_name}: {len(converted_subjects)}/{len(subjects)} subjects mapped")

# Create splits for nnU-Net v2 (only 4 folds)
splits = []
for fold_index in range(4):  # Only folds 0-3 (corresponding to fold1-fold4)
    fold_name = f"fold{fold_index+1}"
    val_subjects = converted_folds[fold_name]
    
    train_subjects = []
    for other_index in range(4):  # Only consider folds 1-4
        other_fold_name = f"fold{other_index+1}"
        if other_index != fold_index:
            train_subjects.extend(converted_folds[other_fold_name])
    
    splits.append({
        "train": train_subjects,
        "val": val_subjects
    })

# Save splits
splits_file = os.path.join(dataset_dir, "splits_final.json")
with open(splits_file, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"✅ splits_final.json created at: {splits_file}")

# Verification
print("\n🔍 Fold summary (4-fold cross-validation):")
for i, split in enumerate(splits):
    print(f"  Fold {i}: {len(split['train'])} train, {len(split['val'])} val")

print(f"\n🎉 DONE! Dataset002 created successfully!")
print(f"📦 Dataset folder: {dataset_dir}")
print(f"🗂️ Mapping file: {mapping_file}")
print(f"📊 Training subjects: {len(label_patient_ids)}, Test subjects: {len(test_new_ids)}")
print(f"🔧 Modality: FA (single channel)")