# Data Directory

This directory defines the structure and instructions for preparing the
datasets used in the **CUPID** retrospective MRI reconstruction experiments.
Prospective imaging data is **not** stored in this repository for size, privacy,
and licensing reasons.

A small number of **example `.npy` files** are provided solely for
demonstration and debugging.

## 📥 1. Downloading the FastMRI Dataset

The dataset used in this project is the **FastMRI knee and brain dataset**  
provided by Facebook AI Research and NYU Langone Health.

You can request access and download the dataset from:

👉 https://fastmri.med.nyu.edu/

**Note:** Users must obtain the full dataset directly from the official FastMRI release
to train the CUPID model in a database manner.

## 🔧 2. Preprocessing Instructions

The provided raw data matrix size for knee data has oversampling in the read-out direction.
These should be removed before feeding the data to CUPID. Similarly, phase-encode lines with no kspace
signal are set to '1' in the retrospective mask. This is handled by the provided code.