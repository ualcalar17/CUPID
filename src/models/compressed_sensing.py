import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from pytorch_wavelets import DWTForward, DWTInverse

from util.sparsity_utils import *
from data.mri_ops import cg_operator, r2c

class soft_thresholding(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def sign_operator(self, Y):
        mask = (abs(Y) < 1e-10)
        Y = Y / abs(Y)
        Y[mask] = 0.0
        return Y
    
    def soft_threshold(self, Yl, Yh, tau_prime):        
        Yl = (torch.abs(Yl) - (tau_prime)).clamp(min=0) * self.sign_operator(Yl)
        Yh[0] = (abs(Yh[0]) - tau_prime).clamp(min=0) * self.sign_operator(Yh[0])
        Yh[1] = (abs(Yh[1]) - tau_prime).clamp(min=0) * self.sign_operator(Yh[1])
        Yh[2] = (abs(Yh[2]) - tau_prime).clamp(min=0) * self.sign_operator(Yh[2])
        Yh[3] = (abs(Yh[3]) - tau_prime).clamp(min=0) * self.sign_operator(Yh[3])
        return Yl, Yh
    
    def forward(self, img, x_pi, levels, wave, tau, wavelet_type='dwt'):        
        if img.ndim < 4:
            for _ in range(4 - img.ndim):
                img = img.unsqueeze(0)
        
        xfm = DWTForward(J=levels, wave=wave, mode='periodization').to(self.device)
        ifm = DWTInverse(wave=wave, mode='periodization').to(self.device)
        x_pi_wav_coeff = wavelet_transform_n_lvl(x_pi, levels, xfm, wavelet_type)
        tau_prime = (abs(x_pi_wav_coeff).max() * tau)
        
        Yl_real, Yh_real = xfm(img.real)
        Yl_imag, Yh_imag = xfm(img.imag)
        Yl = Yl_real + 1j*Yl_imag
        Yh = [Yh_real[i] + 1j * Yh_imag[i] for i in range(levels)]
        
        Yl, Yh = self.soft_threshold(Yl, Yh, tau_prime)
        Yh_real_parts = [Y.real for Y in Yh]
        Yh_imag_parts = [Y.imag for Y in Yh]
        real_img = ifm((Yl.real, Yh_real_parts))
        imag_img = ifm((Yl.imag, Yh_imag_parts))
        return torch.stack((real_img, imag_img), axis=1)

class compressed_sensing(torch.nn.Module):
    def __init__(self, device, levels, wave, wavelet_type, cs_unrolls, cg_mu, cg_iter_cs, tau):
        super().__init__()
        self.st = soft_thresholding(device)
        self.levels = levels
        self.wave = wave
        self.wavelet_type = wavelet_type
        self.cs_unrolls = cs_unrolls
        self.cg_mu = cg_mu
        self.cg_iter_cs = cg_iter_cs
        self.tau = tau

    def forward(self, zerofilled_img, maps, mask, x_PI):
        output = zerofilled_img.clone()
        
        for _ in range(self.cs_unrolls):
            output = cg_operator(zerofilled_img, maps, mask, output, torch.tensor(self.cg_mu), self.cg_iter_cs)
            output = self.st(output, x_PI, self.levels, self.wave, self.tau)
            output = output.squeeze(2)
        return r2c(output,1)