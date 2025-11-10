#!/bin/bash
#BSUB -q gpu
#BSUB -gpu "num=1:j_exclusive=yes:gmem=10.7G"
#BSUB -J nnunet_test
#BSUB -o nnunet_test.%J.out
#BSUB -e nnunet_test.%J.err
#BSUB -L /bin/bash

# ----------------------------------------
# Activate nnU-Net environment
# ----------------------------------------
source /omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/hcp_nnunet_env/bin/activate

# ----------------------------------------
# Set nnU-Net paths (replace with your real directories)
# ----------------------------------------
export nnUNet_raw=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_raw
export nnUNet_preprocessed=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_preprocessed
export nnUNet_results=/omics/groups/OE0441/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/HCP-nnUnetSetup/nnunet_results

# ----------------------------------------
# Debug information
# ----------------------------------------
echo "===== Debug info ====="
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "PATH: $PATH"
echo "Python version: $(python --version)"
echo "nnUNetv2_train path: $(which nnUNetv2_train)"
echo "Check nnU-Net data dirs:"
ls -ld $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# ----------------------------------------
# Test command: show nnUNetv2_train help
# ----------------------------------------
echo "===== Testing nnUNetv2_train ====="
#nnUNetv2_train --help
#nnUNetv2_train 001 3d_fullres 1 --c 
nnUNetv2_train 001 3d_fullres 4 -tr nnUNetTrainerNoDA --c 

echo "===== nnUNetv2_train test completed ====="
