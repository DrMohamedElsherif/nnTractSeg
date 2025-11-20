#!/bin/bash

# Folds to train
FOLDS=(0 1 2 3)

# Paths
ENV_PATH="/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/hcp_nnunet_env/bin/activate"
export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_raw
export nnUNet_preprocessed=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_preprocessed
export nnUNet_results=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_results

# Loop through folds and submit jobs
for FOLD in "${FOLDS[@]}"; do

# ---------------- TRAIN WITH AUGMENTATION (default trainer) ---------------- #
bsub <<EOF
#BSUB -q gpu
#BSUB -gpu "num=1:j_exclusive=yes:gmem=10.7G"
#BSUB -J nnUNet_AUG_f${FOLD}
#BSUB -o nnUNet_AUG_f${FOLD}.%J.out
#BSUB -e nnUNet_AUG_f${FOLD}.%J.err
#BSUB -L /bin/bash

source $ENV_PATH
export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export nnUNet_results=$nnUNet_results

nnUNetv2_train 001 3d_fullres ${FOLD} --c
EOF

# ---------------- TRAIN WITHOUT AUGMENTATION ---------------- #
bsub <<EOF
#BSUB -q gpu
#BSUB -gpu "num=1:j_exclusive=yes:gmem=10.7G"
#BSUB -J nnUNet_NoAUG_f${FOLD}
#BSUB -o nnUNet_NoAUG_f${FOLD}.%J.out
#BSUB -e nnUNet_NoAUG_f${FOLD}.%J.err
#BSUB -L /bin/bash

source $ENV_PATH
export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export nnUNet_results=$nnUNet_results

nnUNetv2_train 001 3d_fullres ${FOLD} -tr nnUNetTrainerNoDA --c
EOF

done

echo "All training jobs submitted! Check with: bjobs"
