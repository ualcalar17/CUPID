# Data Directory

This directory defines the structure and instructions for preparing the datasets used in the **CUPID** retrospective MRI reconstruction experiments.
Prospective imaging data is **not** stored in this repository for size, privacy, and licensing reasons.

A small number of **example `.npy` files** are provided for demonstration and debugging.

## 📥 1. Downloading the FastMRI Dataset

CUPID relies on the **FastMRI knee and brain datasets** developed by Facebook AI Research and NYU Langone Health. These datasets must be downloaded directly from the official FastMRI portal:

👉 https://fastmri.med.nyu.edu/

**Note:** Users must obtain the full dataset directly from the official FastMRI release to train the CUPID model in a database manner (over multiple slices).

## 🔧 2. Preprocessing Instructions

The FastMRI raw k-space data contains oversampling along the readout (frequency-encode) direction. This oversampling must be removed prior to reconstruction.

Similarly, phase-encode lines that contain no signal should be indicated by '1' in the sampling mask. This is done during reconstruction by the handling logic implemented in the provided MRI operator code. Users do not need to modify this behavior unless customizing the sampling pattern.
