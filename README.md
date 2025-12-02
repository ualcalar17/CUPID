# CUPID [NeurIPS '25 Spotlight]
[![arXiv](https://img.shields.io/badge/arXiv-2301.01234-red.svg)](https://arxiv.org/pdf/2411.13022)
[![NeurIPS](https://img.shields.io/badge/Conference-NeurIPS-00BFFF.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/115476)

<p align="center">
  <img src="figs/cover.png" width="900">
</p>

> **Fast MRI for All: Bridging Access Gaps by Training without Raw Data**<br>
> [Yasar Utku Alcalar](https://utkualcalar.github.io/), Merve Gulle, [Mehmet Akcakaya](https://imagine.umn.edu/people/mehmet-akcakaya) <br>
> Neural Information Processing Systems (NeurIPS) 2025
> 
>**Summary**: <br>
>CUPID is a physics-driven deep learning (PD-DL) approach for fast MRI reconstruction that enables high-quality model training using only routine clinical images, **without access to raw k-space data**, making advanced MRI reconstruction more accessible outside specialized centers.

## Setup

### 1. Create Conda Environment and Install Requirements
```bash
conda env create -n cupid python=3.12 -y
conda activate cupid

cd CUPID
pip install requirement.txt
```

### 2. External Libraries Clone the necessary external code for 2D DWT/DCTWT
CUPID relies on advanced wavelet transforms implemented in the `pytorch_wavelets` library. Clone it into your workspace:
```
git clone https://github.com/fbcotter/pytorch_wavelets
```

### 3. Data Preparation
In our retrospective experiments, we used the [fastMRI](https://fastmri.med.nyu.edu/) dataset, acquired with relevant institutional review board approvals. More information regarding this is included in `data/README.md`.

### 4. Zero-Shot Experiments
Once the environment is configured and data is prepared, run zero-shot training and evaluation:
```
bash scripts/train_CUPID_retro.sh
```
The script performs retrospective training using default hyperparameters. Configurations can be customized by modifying YAML files in `configs/` or adjusting shell scripts in `scripts/`.

## 📝 Citation
```bibtex

@inproceedings{alcalar2025cupid,
    title     = {Fast {MRI} for All: Bridging Access Gaps by Training without Raw Data},
    author    = {Yasar Utku Alcalar and Merve Gulle and Mehmet Akcakaya},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year      = {2025}
    url       = {https://openreview.net/forum?id=ugBmWX3H1R}
    }
```
