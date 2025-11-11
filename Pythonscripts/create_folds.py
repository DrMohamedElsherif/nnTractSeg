#!/usr/bin/env python3
import os
import json

# ==== Define folds with original HCP subject IDs (only folds 1-4 for training) ====
folds = {
    "fold1": ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574', '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241', '907656', '904044'],
    "fold2": ['901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579', '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', '859671', '857263', '856766'],
    "fold3": ['849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653', '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370', '771354', '770352', '765056'],
    "fold4": ['761957', '759869', '756055', '753251', '751348', '749361', '748662', '748258', '742549', '734045', '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133', '695768', '690152']
    # fold5 is excluded - used as test set
}

# ==== Paths ====
mapping_file = "/home/m512f/dev/HCP-nnUnetSetup/nnunet_raw/Dataset001_OpticRadiation/patient_id_mapping.txt"
output_file = "/home/m512f/dev/HCP-nnUnetSetup/nnunet_preprocessed/Dataset001_OpticRadiation/splits_final.json"

# ==== Load mapping file ====
original_to_new = {}
new_to_original = {}

with open(mapping_file, 'r') as f:
    for line in f:
        if '->' in line:
            new_id, original_id = line.strip().split('->')
            original_to_new[original_id.strip()] = new_id.strip()
            new_to_original[new_id.strip()] = original_id.strip()

print(f"📋 Loaded mapping for {len(original_to_new)} subjects")

# ==== Verify all fold subjects exist in mapping ====
all_original_subjects_in_folds = set(sum(folds.values(), []))
missing_in_mapping = all_original_subjects_in_folds - set(original_to_new.keys())
if missing_in_mapping:
    print(f"❌ WARNING: {len(missing_in_mapping)} subjects in folds are missing from mapping:")
    for mid in sorted(missing_in_mapping):
        print(f"   - {mid}")
else:
    print("✅ All subjects in folds 1-4 are present in the mapping!")

# ==== Convert folds to new IDs ====
converted_folds = {}
for fold_name, subjects in folds.items():
    converted_subjects = [original_to_new[s] for s in subjects if s in original_to_new]
    converted_folds[fold_name] = converted_subjects
    print(f"📊 {fold_name}: {len(converted_subjects)}/{len(subjects)} subjects mapped")

# ==== Create splits for nnU-Net v2 (only 4 folds) ====
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

# ==== Save splits ====
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\n✅ splits_final.json created at: {output_file}")

# ==== Verification 1: Fold lengths ====
print("\n🔍 Fold summary (4-fold cross-validation):")
for i, split in enumerate(splits):
    print(f"  Fold {i}: {len(split['train'])} train, {len(split['val'])} val")

# ==== Verification 2: Check all mapped training subjects included ====
all_training_subjects_in_splits = set()
for split in splits:
    all_training_subjects_in_splits.update(split['train'])
    all_training_subjects_in_splits.update(split['val'])

# Get all subjects that should be in training (everything except test/fold5)
all_training_original_subjects = set(sum(folds.values(), []))
all_training_new_subjects = set(original_to_new[s] for s in all_training_original_subjects if s in original_to_new)

missing_in_splits = all_training_new_subjects - all_training_subjects_in_splits
if missing_in_splits:
    print(f"❌ {len(missing_in_splits)} training subjects missing from splits: {missing_in_splits}")
else:
    print("✅ All training subjects included in splits")

# ==== Verification 3: Check train/val overlap per fold ====
for i, split in enumerate(splits):
    overlap = set(split['train']) & set(split['val'])
    if overlap:
        print(f"❌ Fold {i}: overlap between train and val: {overlap}")
    else:
        print(f"✅ Fold {i}: no overlap between train and val")

# ==== Verification 4: Check all original fold subjects included in final JSON ====
all_original_training_subjects = set(sum(folds.values(), []))
all_converted_subjects_in_splits = set()
for split in splits:
    all_converted_subjects_in_splits.update(split['train'])
    all_converted_subjects_in_splits.update(split['val'])

# Map converted IDs back to original to compare
converted_back_to_original = {v: k for k, v in original_to_new.items()}
subjects_in_json_as_original = {converted_back_to_original[s] for s in all_converted_subjects_in_splits}

missing_original_subjects = all_original_training_subjects - subjects_in_json_as_original

if missing_original_subjects:
    print(f"❌ WARNING: {len(missing_original_subjects)} original training subjects missing in final JSON:")
    for sub in sorted(missing_original_subjects):
        print(f"   - {sub}")
else:
    print("✅ All original training subjects (folds 1-4) are included in the final JSON file")

print(f"\n📝 Note: Fold5 subjects are reserved for testing and are not included in these splits")
# ==== End of script ====