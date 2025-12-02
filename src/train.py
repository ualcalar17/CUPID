import os
import yaml
import argparse
import torch
import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perturbations import *
from data.mri_ops import *
from data.loader import *
from util.eval_utils import *
from util.sparsity_utils import wavelet_loss_function
from models.unrolled_net import UnrolledNet
from src.models.compressed_sensing import compressed_sensing
from pathlib import Path

def load_yaml(file_path):
    file_path = Path(file_path)
    with open(file_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def main(args):    
    data_config, mask_config = load_config_for_dataset('configs/data_config.yaml')
    model_config = load_yaml("configs/model_config.yaml")
    cs_config = load_yaml("configs/sparsity_config.yaml")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = UnrolledNet(device, **model_config).to(device)
    cs = compressed_sensing(device, **cs_config.get('CS'))
    ksp_loader, maps_loader, mask = load_data(device, **data_config, **mask_config)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    print('*' * 60 + '\nSTARTED Training with R = \"%s\", ' % mask_config.get('acc_rate') 
        + 'epochs = \"%s\", ' %args.epochs + 'LR = \"%s\", ' %args.learning_rate + 'lambda = \"%s\" \n' %args.lambda_val)
    train_loss_epoch, mu_values = [], []
    x_m_full = torch.zeros((1,300,320,368), dtype=torch.complex64).to(device)
    
    for m in range (args.nb_rewightings):
        for epoch in tqdm.tqdm(range(args.epochs), ncols=60):
            train_slice_loss = []
            for slice_num, (ksp_batch, maps_batch) in enumerate(zip(ksp_loader, maps_loader)):
                
                if epoch == 0: x_PI, ref_img = get_x_PI(ksp_batch, maps_batch, mask)
                else: pass
                zerofilled_img = EHE(x_PI, maps_batch, mask)
                rssq_maps = torch.sqrt(torch.sum(maps_batch**2, dim=1))
                
                ##################### PARALLEL IMAGING FIDELITY PART #####################
                unperturbed_output, mu = model(c2r(zerofilled_img, dim=1), maps_batch, mask)
                unperturbed_output = r2c(unperturbed_output, dim=1)
                
                total_loss = 0
                for k in range(args.nb_perturbations):
                    p_k = torch.zeros_like(x_PI, dtype = torch.complex64)                     
                    p_k = process_p_k(p_k, k, args.nb_perturbations).to(device)
                    EHE_pk = EHE(p_k, maps_batch, mask)
                    perturbed_input = zerofilled_img + EHE_pk
                    perturbed_output, _ = model(c2r(perturbed_input, dim=1), maps_batch, mask)
                    perturbed_output = r2c(perturbed_output, dim=1)
                    l_pif = torch.norm((perturbed_output - unperturbed_output) - (p_k * (abs(rssq_maps)!=0)), p=2) / torch.norm(p_k*(abs(rssq_maps)!=0), p=2)
                    total_loss += l_pif            
                total_loss *= (args.lambda_val/args.nb_perturbations)
                ##########################################################################
                
                ##################### COMPRESSIBILITY PART #####################
                unperturbed_output = unperturbed_output * (abs(rssq_maps)!=0)
                
                if epoch == 0 and m == 0:
                    x_m = cs(c2r(zerofilled_img, dim=1), maps_batch, mask, x_PI)
                    x_m_full[:,slice_num,:,:] = x_m
                elif epoch == 0 and m > 0:
                    x_m = unperturbed_output.clone().detach()
                    x_m_full[:,slice_num,:,:] = x_m
                else:
                    pass
                lcomp_selected_keys = {key: cs_config['CS'][key] for key in ['levels', 'wave', 'wavelet_type']}
                l_comp = wavelet_loss_function(device, x_m_full[:,slice_num,:,:], unperturbed_output, args.eps, **lcomp_selected_keys)
                ################################################################
                
                total_loss += l_comp
                train_slice_loss.append(total_loss)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            mu_values.append(mu.detach().cpu().numpy())
            train_loss_epoch.append(np.mean(torch.tensor(train_slice_loss).detach().numpy()))
            
            if epoch % args.save_freq == 0 or epoch == args.epochs-1:
                torch.save(model.state_dict(), args.out_path+'/weights/CUPID_model_weights_at_epoch{:04}.pth'.format(((m * args.epochs) + epoch)))
                psnr_val = cal_PSNR(clear_data(ref_img), clear_data(unperturbed_output))     
                ssim_val = cal_SSIM(clear_data(ref_img), clear_data(unperturbed_output))          
                print(f'\n\nPSNR value --> {psnr_val:.2f}')
                print(f'SSIM value --> {ssim_val:.3f}')
                save_imgs(args.out_path, ref_img, perturbed_input, perturbed_output, unperturbed_output, x_PI, zerofilled_img, p_k, x_m, mask, m*args.epochs+epoch)
                plt.imshow(np.flipud(clear_data(unperturbed_output)), cmap='gray', vmax=1.0)
                plt.text(7, 313, f'PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}', color='white', fontsize=12, fontweight='bold', ha='left', va='bottom')
                plt.axis('off')
                plt.savefig(args.out_path + f'recon/progress/recon_{(m*args.epochs+epoch):03}_with_metrics.png', bbox_inches='tight', pad_inches=0)
                plt.close('all')
        print('\n w_k^{%s}' % m + ' changed to w_k^{%s}' % (m+1) + '\n')
    plot_loss(args.out_path, train_loss_epoch, args.epochs*(args.nb_rewightings))
    plot_mu(args.out_path, mu_values, args.epochs*(args.nb_rewightings))
    print('\nFINISHED Training!\n' + '*' * 60)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path',
        type=str,
        default='results/pd_01/',
        help='results file directory'
    )
    parser.add_argument(
        '--gpu_id',
        type=str,
        default='0',
        help='which GPU to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-4,
        help='epsilon for the loss function'
    )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=20,
        help='result saving frequency'
    )
    parser.add_argument(
        '--nb_rewightings',
        type=int,
        default=1,
        help='number of reweightings for l_comp denominator term'
    )
    parser.add_argument(
        '--nb_perturbations',
        type=int,
        default=6,
        help='number of perturbations'
    )
    parser.add_argument(
        '--lambda_val',
        type=float,
        default=200.0,
        help='trade-off parameter for the loss function'
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_path+'inputs/'): os.makedirs(args.out_path+'inputs/')
    if not os.path.exists(args.out_path+'loss_related/'): os.makedirs(args.out_path+'loss_related/')
    if not os.path.exists(args.out_path+'recon/progress/'): os.makedirs(args.out_path+'recon/progress/')
    if not os.path.exists(args.out_path+'weights/'): os.makedirs(args.out_path+'weights/')
    main(args)