import torch
from pytorch_wavelets import DWTForward, DTCWTForward

def wavelet_transform_n_lvl(img, levels, xfm, wavelet_type):
    def concat_4(top_left, top_right, bttm_left, bttm_right):
        top_row = torch.cat((top_left, top_right), dim=3)
        bottom_row = torch.cat((bttm_left, bttm_right), dim=3)
        return torch.cat((top_row, bottom_row), dim=2)
    
    if img.ndim < 4:
        for _ in range(4 - img.ndim):
            img = img.unsqueeze(0)
    
    Yl_real, Yh_real = xfm(img.real)
    Yl_imag, Yh_imag = xfm(img.imag)
    Yl = Yl_real + 1j*Yl_imag
    Yh = [Yh_real[i] + 1j * Yh_imag[i] for i in range(levels)]
    
    if wavelet_type == 'dwt':
        concat_img = Yl
        for i in range(levels-1, -1, -1):
            concat_img = concat_4(concat_img, Yh[i][:, :, 0, ...], Yh[i][:, :, 1, ...], Yh[i][:, :, 2, ...])
        return concat_img
    if wavelet_type == 'dctwt':
        return Yl, Yh

def wavelet_loss_function(device, output_prev, output, eps, levels, wave, wavelet_type):
    assert levels >= 1, "Levels can not be smaller than 1."
    out_prev = output_prev.clone()
    out = output.clone()
    
    def wavelet_loss(out_prev, out, wave_type):
        xfm = DWTForward(J=levels, wave=wave_type, mode='periodization').to(device)   
        out_prev_wavelet_coeffs = wavelet_transform_n_lvl(out_prev, levels, xfm, wavelet_type)
        out_wavelet_coeffs = wavelet_transform_n_lvl(out, levels, xfm, wavelet_type)
        return (torch.sum(abs(out_wavelet_coeffs) / (abs(out_prev_wavelet_coeffs) + eps))) / (out_prev_wavelet_coeffs.size(-1)*out_prev_wavelet_coeffs.size(-2))
    
    if wavelet_type == 'dwt':        
        return wavelet_loss(out_prev, out, wave)
        
    if wavelet_type == 'dctwt':
        def loss_1lvl(out_prev_wavelet_coeffs, out_wavelet_coeffs):
            return (torch.sum(abs(out_wavelet_coeffs) / (abs(out_prev_wavelet_coeffs) + eps))) / (out_prev_wavelet_coeffs.size(-1)*out_prev_wavelet_coeffs.size(-2))
        def loss_1lvl_yh(out_prev_wavelet_coeffs, out_wavelet_coeffs):
            return (torch.sum(abs(out_wavelet_coeffs) / (abs(out_prev_wavelet_coeffs) + eps))) / (out_prev_wavelet_coeffs.size(-2)*out_prev_wavelet_coeffs.size(-3))
        xfm = DTCWTForward(J=levels, biort='near_sym_b', qshift='qshift_b').to(device)
        out_prev_Yl, out_prev_Yh = wavelet_transform_n_lvl(out_prev, levels, xfm, wavelet_type)
        out_Yl, out_Yh = wavelet_transform_n_lvl(out, levels, xfm, wavelet_type)
        return (loss_1lvl(out_prev_Yl, out_Yl) + loss_1lvl_yh(out_prev_Yh[0], out_Yh[0]) + loss_1lvl_yh(out_prev_Yh[1], out_Yh[1]) + loss_1lvl_yh(out_prev_Yh[2], out_Yh[2]) + loss_1lvl_yh(out_prev_Yh[3], out_Yh[3]))/5