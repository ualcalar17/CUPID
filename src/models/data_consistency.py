import torch
from data.mri_ops import c2r, r2c, EHE

class Data_consistency(torch.nn.Module):
    """
    x = (E^h*E + mu*I)^-1 (E^h*y + mu*z)
    """
    def __init__(self, CG_Iter, mu):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(mu), requires_grad = True)
        self.cg_iter = CG_Iter

    def forward(self, zerofilled, coil, mask, denoiser_out):
        r_now = r2c(zerofilled, 1) + self.mu * r2c(denoiser_out, 1) # E^h*y + mu*z = p
        p_now = torch.clone(r_now)
        b_approx = torch.zeros_like(p_now)

        for _ in range(self.cg_iter):
            q = EHE(p_now, coil, mask) + self.mu * p_now # A * p = (E^h*E + mu*I) * p = E^hE(p) + mu*p
            alpha = torch.sum(r_now*torch.conj(r_now)) / torch.sum(q*torch.conj(p_now))
            b_next = b_approx + alpha*p_now
            r_next = r_now - alpha*q
            p_next = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now
            b_approx = b_next
    
            p_now = torch.clone(p_next)
            r_now = torch.clone(r_next)

        return c2r(b_approx, 1), self.mu