#!/bin/bash

# ==== Activate your environment ====
source ~/dev/hcp_nnunet_env/bin/activate

# ==== Set nnU-Net paths ====
export nnUNet_raw=/home/m512f/dev/HCP-nnUnetSetup/nnunet_raw
export nnUNet_preprocessed=/home/m512f/dev/HCP-nnUnetSetup/nnunet_preprocessed
export nnUNet_results=/home/m512f/dev/HCP-nnUnetSetup/nnunet_results

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"

# ==== Find the latest dataset folder ====
LATEST_DATASET=$(ls -d $nnUNet_raw/Dataset*_OpticRadiation 2>/dev/null | sort | tail -n 1)

if [ -z "$LATEST_DATASET" ]; then
    echo "❌ No dataset found in $nnUNet_raw"
    exit 1
fi

DATASET_NAME=$(basename "$LATEST_DATASET")
echo "✅ Latest dataset detected: $DATASET_NAME"

# ==== Extract dataset numeric ID for nnUNetv2_plan_and_preprocess ====
# Dataset names are like Dataset001_OpticRadiation
DATASET_ID=$(echo "$DATASET_NAME" | grep -oP '(?<=Dataset)\d{3}' | sed 's/^0*//')
if [ -z "$DATASET_ID" ]; then
    echo "❌ Could not extract dataset ID from $DATASET_NAME"
    exit 1
fi
echo "🔢 Dataset numeric ID: $DATASET_ID"

# ==== Run preprocessing using nnUNetv2_plan_and_preprocess ====
echo "⏳ Running nnU-Net v2 preprocessing..."
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

if [ $? -eq 0 ]; then
    echo "✅ Preprocessing complete for $DATASET_NAME"
else
    echo "❌ Preprocessing failed for $DATASET_NAME"
    exit 1
fi
