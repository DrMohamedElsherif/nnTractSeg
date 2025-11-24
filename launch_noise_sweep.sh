#!/bin/bash

# ===============================================================
# TESTING SCRIPT - SINGLE FOLD ONLY
#  - all 12 noise-variance trainers (UL1 → H2)
#  - ONLY fold 2 for testing
#
# Total jobs: 12 
# ===============================================================

TRAINERS=(
    nnUNetTrainerPeaksDA_UL1
    nnUNetTrainerPeaksDA_UL2
    nnUNetTrainerPeaksDA_UL3
    nnUNetTrainerPeaksDA_UL4
    nnUNetTrainerPeaksDA_L1
    nnUNetTrainerPeaksDA_L2
    nnUNetTrainerPeaksDA_L3
    nnUNetTrainerPeaksDA_L4
    nnUNetTrainerPeaksDA_L5
    nnUNetTrainerPeaksDA_M1
    nnUNetTrainerPeaksDA_M2
    nnUNetTrainerPeaksDA_H1
    nnUNetTrainerPeaksDA_H2
)

DATASET=001
CONFIG=3d_fullres

# CHANGE: Only fold 2 for testing
for TR in "${TRAINERS[@]}"; do
    for FOLD in 2; do  # ONLY FOLD 2
        
        JOB_NAME="${TR}_F${FOLD}"

        bsub <<EOT
#BSUB -q gpu
#BSUB -gpu "num=1:j_exclusive=yes:gmem=10.7G"
#BSUB -J ${JOB_NAME}
#BSUB -o ${JOB_NAME}.%J.out
#BSUB -e ${JOB_NAME}.%J.err
#BSUB -L /bin/bash

source /omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/hcp_nnunet_env/bin/activate

export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_raw
export nnUNet_preprocessed=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_preprocessed
export nnUNet_results=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_results

echo "Running trainer: ${TR}  Fold: ${FOLD}"

nnUNetv2_train $DATASET $CONFIG $FOLD -tr $TR --c

EOT

    done
done