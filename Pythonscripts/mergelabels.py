# mergelabels.py

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import argparse

def merge_OR_labels(patient_folder, output_folder, binary=False):
    os.makedirs(output_folder, exist_ok=True)
    pid = os.path.basename(patient_folder)
    tracts = os.path.join(patient_folder, "tracts")
    left = os.path.join(tracts, "OR_left.nii.gz")
    right = os.path.join(tracts, "OR_right.nii.gz")

    missing = []
    if not os.path.exists(left): missing.append("left")
    if not os.path.exists(right): missing.append("right")
    if missing: return False, missing, None

    left_img, right_img = nib.load(left), nib.load(right)
    left_data, right_data = left_img.get_fdata(), right_img.get_fdata()

    merged = np.zeros_like(left_data, dtype=np.uint8)
    if binary:
        merged = np.maximum(left_data, right_data).astype(np.uint8)
    else:
        merged[left_data > 0] = 1
        merged[right_data > 0] = 2

    temp_path = os.path.join(output_folder, f"{pid}.nii.gz")
    nib.save(nib.Nifti1Image(merged, left_img.affine, left_img.header), temp_path)
    return True, [], temp_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--mapping_file", required=True)
    args = parser.parse_args()

    # Load mapping
    mapping = {}
    if os.path.exists(args.mapping_file):
        with open(args.mapping_file) as f:
            for line in f:
                if "->" in line:
                    num, orig = line.strip().split("->")
                    mapping[orig.strip()] = num.strip()

    patients = [
        os.path.join(args.parent_folder, p)
        for p in os.listdir(args.parent_folder)
        if os.path.isdir(os.path.join(args.parent_folder, p))
    ]

    skipped = {}
    for p in tqdm(patients, desc="Merging OR labels", unit="patient"):
        success, missing, temp = merge_OR_labels(p, args.output_folder, binary=False)
        pid = os.path.basename(p)
        if not success:
            skipped[pid] = missing
            continue

        if pid in mapping:
            new_name = f"{mapping[pid]}.nii.gz"
            new_path = os.path.join(args.output_folder, new_name)
            os.rename(temp, new_path)

    print(f"\n✅ Merging complete! {len(patients)-len(skipped)} / {len(patients)} processed.")
    if skipped:
        for k,v in skipped.items():
            print(f"⚠️ Missing {v} for {k}")
