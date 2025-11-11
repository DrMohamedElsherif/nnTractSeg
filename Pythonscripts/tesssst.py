import os
import json
import re
import argparse
import sys

# ====== Hardcoded cross-validation folds (HCP IDs) ======
folds = {
    "fold1": ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574', '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241', '907656', '904044'],
    "fold2": ['901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579', '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', '859671', '857263', '856766'],
    "fold3": ['849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653', '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370', '771354', '770352', '765056'],
    "fold4": ['761957', '759869', '756055', '753251', '751348', '749361', '748662', '748258', '742549', '734045', '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133', '695768', '690152'],
    "fold5": ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']
}


def find_latest_dataset(nnunet_raw_path):
    """Find the most recent DatasetXXX_OpticRadiation folder."""
    pattern = re.compile(r"Dataset\d{3}_OpticRadiation$")
    datasets = [
        os.path.join(nnunet_raw_path, d)
        for d in os.listdir(nnunet_raw_path)
        if pattern.match(d)
    ]
    if not datasets:
        print(f"❌ No DatasetXXX_OpticRadiation folders found in {nnunet_raw_path}")
        sys.exit(1)
    latest = sorted(datasets)[-1]
    print(f"✅ Latest dataset detected: {os.path.basename(latest)}")
    return latest


def load_mapping(mapping_file):
    """Return dict: original_id -> numeric_id"""
    mapping = {}
    with open(mapping_file) as f:
        for line in f:
            if "->" in line:
                num, orig = line.strip().split("->")
                mapping[orig.strip()] = num.strip()
    return mapping


def create_splits(mapping, output_path):
    """Create nnU-Net standard splits_final.json"""
    all_ids = list(mapping.values())
    splits = []

    for fold_name, val_originals in folds.items():
        val_ids = [mapping[o] for o in val_originals if o in mapping]
        train_ids = [pid for pid in all_ids if pid not in val_ids]
        splits.append({
            "train": train_ids,
            "val": val_ids
        })

    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\n✅ Created nnU-Net splits file: {output_path}")
    for i, s in enumerate(splits, start=1):
        print(f"  Fold {i}: {len(s['train'])} train / {len(s['val'])} val")

    # --- Show example content to verify mapping ---
    print("\n🔍 Example fold content (showing first 5 val subjects with mapping):")
    for i, (fold_name, val_originals) in enumerate(folds.items(), start=1):
        print(f"\n📁 Fold {i} ({fold_name}) examples:")
        shown = 0
        for orig in val_originals:
            if orig in mapping:
                mapped = mapping[orig]
                print(f"   HCP {orig}  →  nnU-Net ID {mapped}")
                shown += 1
            if shown >= 5:  # show only first 5 examples
                break
        if shown == 0:
            print("   ⚠️ No matching subjects from mapping (check your mapping file!)")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create nnU-Net-compatible splits_final.json (auto-detect latest dataset)")
    parser.add_argument("--nnunet_raw", default="/home/m512f/dev/HCP-nnUnetSetup/nnunet_raw",
                        help="Base nnUNet_raw folder")
    parser.add_argument("--nnunet_preprocessed", default="/home/m512f/dev/HCP-nnUnetSetup/nnunet_preprocessed",
                        help="Base nnUNet_preprocessed folder")
    args = parser.parse_args()

    # Find dataset folder in raw
    raw_dataset_dir = find_latest_dataset(args.nnunet_raw)

    # Get dataset folder name (e.g., Dataset001_OpticRadiation)
    dataset_name = os.path.basename(raw_dataset_dir)

    # Create target directory path in nnUNet_preprocessed
    preprocessed_dataset_dir = os.path.join(args.nnunet_preprocessed, dataset_name)

    # Final save path (changed here!)
    output_file = os.path.join(preprocessed_dataset_dir, "splits_final.json")

    mapping_file = os.path.join(raw_dataset_dir, "patient_id_mapping.txt")
    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file not found in {raw_dataset_dir}")
        sys.exit(1)

    mapping = load_mapping(mapping_file)
    create_splits(mapping, output_file)

    print(f"\n🎉 Splitting complete!")
    print(f"📌 Saved to: {output_file}")
