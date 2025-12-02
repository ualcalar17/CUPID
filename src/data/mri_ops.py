import torch

def E(image, coil, mask):
    """Applies forward encoding operation
    
    Args:
        image (torch.Tensor): The input image tensor of shape (B, 1, H, W).
        coil (torch.Tensor): The coil sensitivity maps of shape (B, C, H, W).
        mask (torch.Tensor): The undersampling mask of shape (H, W).

    Returns:
        torch.Tensor: The encoded k-space data of shape (B, C, H, W).
    """
    image = image.repeat(coil.shape[0],1,1)
    return fft2c(image*coil, [2,3]) * mask[None,None,...]
    
def EH(kspace, coil, mask):
    """Applies adjoint encoding operation
    
    Args:
        kspace (torch.Tensor): The acquired k-space data of shape (B, C, H, W).
        coil (torch.Tensor): The coil sensitivity maps of shape (B, C, H, W).
        mask (torch.Tensor): The undersampling mask of shape (H, W).

    Returns:
        torch.Tensor: The reconstructed image of shape (B, H, W).
    """
    return torch.sum(ifft2c(kspace*mask[None,None,...], [2,3]) * torch.conj(coil), dim=1)

def EHE(image, coil, mask):
    """Normal operator (E^H * E)"""
    return EH(E(image, coil, mask), coil, mask) 

def fft2c(kspace, dims):
    """Applies centered Fast Fourier Transform (FFT).
    
    Args:
        kspace (torch.Tensor): Input tensor representing k-space data.
        dims (int or tuple of ints): Dimensions along which to apply the FFT.
    
    Returns:
        torch.Tensor: FFT of the input with orthogonal normalization.
    """
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(kspace, dim=dims), dim=dims, norm="ortho"),
        dim=dims,
    )

def ifft2c(img, dims):
    """Applies centered Inverse Fast Fourier Transform (IFFT).
    
    Args:
        img (torch.Tensor): Input tensor in the Fourier domain.
        dims (int or tuple of ints): Dimensions along which to apply the IFFT.
    
    Returns:
        torch.Tensor: IFFT of the input with orthogonal normalization.
    """
    return torch.fft.ifftshift(
        torch.fft.ifftn(torch.fft.fftshift(img, dim=dims), dim=dims, norm="ortho"),
        dim=dims,
    )

def r2c(img, dim):
    """Convert real-valued tensor representation to complex tensor.
    
    Args:
        img (torch.Tensor): Input tensor with real and imaginary parts stacked along `ax`.
        ax (int): The axis along which real and imaginary parts are stacked.
        
    Returns:
        torch.Tensor: Complex-valued tensor.
    """
    assert img.shape[dim] == 2, f"Expected dimension {dim} to have size 2, but got shape {img.shape}"
    return torch.complex(img.select(dim=dim, index=0), img.select(dim=dim, index=1))

def c2r(img, dim):
    """Convert complex tensor to real-valued representation.
    
    Args:
        img (torch.Tensor): Complex-valued tensor.
        ax (int): The axis along which to stack real and imaginary parts.
        
    Returns:
        torch.Tensor: Tensor with real and imaginary parts stacked along `ax`.
    """
    return torch.stack((img.real, img.imag), dim=dim)

    
def cg_operator(zerofilled, coil, mask, output, mu, nb_iters):
    '''Conjugate Gradient (CG) solver for regularized MRI reconstruction.

    Args:
        zerofilled (torch.Tensor): The zero-filled reconstruction (initial estimate) of shape (B, H, W).
        coil (torch.Tensor): The coil sensitivity maps of shape (C, H, W).
        mask (torch.Tensor): The undersampling mask of shape (H, W).
        output (torch.Tensor): The previous reconstruction estimate of shape (B, H, W).
        mu (float): Regularization parameter controlling data fidelity.
        nb_iters (int): Number of conjugate gradient iterations.

    Returns:
        torch.Tensor: The estimated image after CG iterations, shape (B, H, W).
    '''
    r_now = r2c(zerofilled, 1) + mu * r2c(output, 1)
    p_now = torch.clone(r_now)
    b_approx = torch.zeros_like(p_now)
    for _ in range(nb_iters):
        q = EHE(p_now, coil, mask) + mu * p_now
        alpha = torch.sum(r_now*torch.conj(r_now)) / (torch.sum(q * torch.conj(p_now)) + 1e-20)
        b_next = b_approx + alpha*p_now
        r_next = r_now - alpha*q
        p_next = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now
        b_approx = b_next
        p_now = torch.clone(p_next)
        r_now = torch.clone(r_next)
    return b_approx