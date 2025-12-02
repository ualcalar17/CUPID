import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

def cal_SSIM(ref, recon):
    return structural_similarity(ref, recon, data_range=recon.max() - recon.min())

def cal_PSNR(ref, recon):
    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    return 20 * np.log10(ref.max() / (np.sqrt(mse) + 1e-10))

def norm_to_01(image):
    '''Normalizes an image to the [0,1] range'''
    return (image - image.min()) / (image.max() - image.min())

def save_img(path, img, max_v = 1.0, flip= True):
    '''Preprocesses the image and saves it as a grayscale image at the specified path'''
    img = clear_data(img)
    if flip == True:
        img = np.flipud(img)
    plt.imsave(path, norm_to_01(img), cmap='gray', vmax = max_v)

def clear_data(img, take_abs=True):
    '''Preprocesses the image and saves it as a grayscale image at the specified path'''
    if isinstance(img, torch.Tensor):
        if img.requires_grad:  
            img = img.detach()
        img = img.cpu().numpy()
    img = img.squeeze()
    if take_abs == True:
        img = abs(img)
    return img

def plot_loss(root, loss, epochs):
    plt.close('all')
    plt.plot(range(1, epochs+1), loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Epochs')
    plt.grid(True)
    plt.savefig(root + 'train_loss_vs_epochs.png')

def plot_mu(root, mu_values, epochs):
    plt.close('all')
    plt.plot(range(1, epochs+1), mu_values)
    plt.xlabel('Epochs')
    plt.ylabel('mu')
    plt.title('mu vs Epochs')
    plt.grid(True)
    plt.savefig(root + 'mu_vs_epochs.png')

def save_imgs(root, ref, perturbed_input, perturbed_output, recon, x_PI, x_zf, p_k, x_m, mask, epoch, intensity=1.0):
    save_img(root + 'inputs/ref_img.png', ref, intensity)
    save_img(root + 'inputs/x_PI.png', x_PI, intensity)
    save_img(root + 'inputs/zf_img.png', x_zf, intensity)
    plt.imsave(root + 'inputs/mask.png', clear_data(mask), cmap='gray')
    save_img(root + 'inputs/perturbed_input.png', perturbed_input, intensity)
    
    save_img(root + 'loss_related/p_k.png', p_k, intensity)
    save_img(root + 'loss_related/p_k_est.png', (perturbed_output - recon), intensity)
    save_img(root + 'loss_related/x_m.png', x_m, intensity)
    
    save_img(root + 'recon/perturbed_output.png', perturbed_output, intensity)   
    save_img(root + 'recon/recon.png', recon, intensity)