# splitpeaks.py

import os
import nibabel as nib
from tqdm import tqdm
import argparse

def split_4d_nifti_one_patient(patient_folder, output_folder, patient_enum):
    os.makedirs(output_folder, exist_ok=True)
    input_path = os.path.join(patient_folder, 'peaks.nii.gz')
    patient_id = os.path.basename(patient_folder)
    
    if not os.path.isfile(input_path):
        print(f"No peaks.nii.gz found in {patient_folder}")
        return False
    
    img = nib.load(input_path)
    data = img.get_fdata()
    if data.ndim != 4:
        print(f"Image is not 4D but {data.ndim}D in {patient_id}")
        return False
    
    for t in range(data.shape[3]):
        volume_3d = data[..., t]
        new_img = nib.Nifti1Image(volume_3d, affine=img.affine, header=img.header)
        filename = f"{patient_enum:03d}_000{t}.nii.gz"
        nib.save(new_img, os.path.join(output_folder, filename))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--mapping_file", required=True)
    args = parser.parse_args()

    patient_folders = sorted([
        os.path.join(args.parent_folder, f)
        for f in os.listdir(args.parent_folder)
        if os.path.isdir(os.path.join(args.parent_folder, f))
    ])

    mapping_lines = []
    for i, patient_path in enumerate(tqdm(patient_folders, desc="Splitting peaks", unit="patient"), start=1):
        success = split_4d_nifti_one_patient(patient_path, args.output_folder, i)
        if success:
            original_id = os.path.basename(patient_path)
            mapping_lines.append(f"{i:03d} -> {original_id}\n")

    os.makedirs(os.path.dirname(args.mapping_file), exist_ok=True)
    with open(args.mapping_file, "w") as f:
        f.writelines(mapping_lines)

    print(f"\n✅ Finished splitting {len(mapping_lines)} patients into 3D volumes.")
    print(f"🗂️ Mapping file saved at: {args.mapping_file}")
