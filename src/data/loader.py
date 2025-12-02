import torch
import numpy as np
import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.mri_ops import c2r, EH, ifft2c, cg_operator

def load_config_for_dataset(yaml_file):
    """Loads dataset-specific configuration from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    dataset_type = config.get('dataset_type')
    if dataset_type in config:
        return config[dataset_type], config.get('mask', {})
    else:
        raise ValueError(f"Error: Dataset type '{dataset_type}' not found in the config.")

def generate_mask(ksp, ACS_length, acc_rate, mask_type):
    """Generates a boolean mask where k-space values are zero or based on undersampling."""
    _, _, Nro, Npe = ksp.size()
    mask = torch.zeros(size=(Nro, Npe), dtype=torch.bool, device=ksp.device)
    
    # Set True wherever ksp has no signal
    mask[abs(ksp[0,0,...]) < 5e-9] = True
    
    ACS_start = int((Npe - ACS_length) / 2) - 1
    ACS_end = ACS_start + ACS_length
    mask[:, ACS_start:ACS_end] = True
    
    if mask_type == 'random':
        import random
        random.seed(41)
        num_sampled_points = int(Npe / acc_rate) - ACS_length
        sampled_indices = random.sample([i for i in range(Npe) if i < ACS_start or i >= ACS_end], num_sampled_points)
        mask[:, sampled_indices] = True
    if mask_type == 'equidistant':
        mask[:, 0::acc_rate] = True
    return mask

def load_data(device, data_path, **mask_config):
    data = np.load(data_path, allow_pickle=True).item()
    kspace_train, coils_train = data['kspace'], data['coils']
    ksp_tensor = torch.from_numpy(kspace_train).to(torch.complex64).to(device).permute(0,3,1,2) #[B, H, W, C] --> [B, C, H, W]
    coils_tensor = torch.from_numpy(coils_train).to(torch.complex64).to(device).permute(0,3,1,2) #[B, H, W, C] --> [B, C, H, W]
    mask = generate_mask(ksp_tensor, **mask_config)
    ksp_loader = torch.utils.data.DataLoader(ksp_tensor, batch_size=1, shuffle=False)
    coils_loader = torch.utils.data.DataLoader(coils_tensor, batch_size=1, shuffle=False)
    return ksp_loader, coils_loader, mask

def get_x_PI(ksp_batch, maps_batch, mask):
    normalized_ksp = ksp_batch / torch.max(torch.abs(ksp_batch))
    ref_img = torch.sum(torch.conj(maps_batch) * ifft2c(normalized_ksp, [2,3]), dim=1) #[B, H, W]
    zerofilled_ksp = normalized_ksp*mask #[B, C, H, W]
    zerofilled_img = EH(zerofilled_ksp, maps_batch, mask) #[B, H, W]
    x_PI = cg_operator(c2r(zerofilled_img, 1), maps_batch, mask, torch.zeros_like(c2r(zerofilled_img, 1)), torch.tensor(0.0), 30)
    return x_PI, ref_img