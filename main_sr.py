import logging
import os

import numpy as np
import torch
from torchvision.transforms.functional import resize
from collections import OrderedDict

from utils import utils_function
from utils.utils_svd import SuperResolution
from utils.utils_inverse import generation_steps

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name              = '256x256_diffusion_uncond'      # ffhq_diffusion | 256x256_diffusion_uncond
    testset_name            = 'imagenet_test'                  # 'ffhq_test' | 'imagenet_val' | 'ffhq_val' | 'imagenet_test'
    num_train_timesteps     = 1000
    iter_num                = 20                              # set number of iterations
    skip                    = num_train_timesteps//iter_num   # skip interval

    save_L                  = True              # save degraded image
    save_E                  = True              # save estimated image
    
    sigma_0                 = 0.05
    eta                     = 0.85
    etaB                    = 1
	
    generate_mode           = 'DDRM'
    skip_type               = 'quad'            # uniform | quad   
    sf                      = 4                 # set scale factor
    sr_mode                 = 'blur'            # blur | cubic
    border                  = sf
         
    n_channels              = 3
    img_size                = 256
    cwd                     = ''  
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results')      # fixed
    result_name             = f'{testset_name}_{sr_mode}_sr{sf}_{generate_mode}_{model_name}_sigma{sigma_0}_NFE{iter_num}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # noise schedule 
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    betas                   = torch.cat([torch.zeros(1).to(device), betas], dim=0)
    
    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name)   # L_path, for original images
    E_path = os.path.join(results, result_name)     # E_path, for result images
    if not os.path.exists(E_path):
        os.makedirs(E_path)

    logger_name = result_name
    utils_function.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'ffhq_diffusion' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_function.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    model = model.to(device)

    logger.info('model_name:{}'.format(model_name))
    logger.info('skip_type:{}, skip interval:{}'.format(skip_type, skip))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = utils_function.get_image_paths(L_path)


    def running():
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []

        for idx, img in enumerate(L_paths):
            
            img_name, ext = os.path.splitext(os.path.basename(img))
            np.random.seed(seed=0)  # for reproducibility

            # --------------------------------
            # (1) get original images
            # --------------------------------

            img_H = utils_function.imread_uint(img, n_channels=n_channels)

            img_H_tensor = np.transpose(img_H, (2, 0, 1))
            img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
            img_H_tensor = resize(img_H_tensor, size=(img_size, img_size))
            img_H_tensor = img_H_tensor / 255 * 2 - 1.0

            # --------------------------------
            # (2) apply degradation
            # --------------------------------

            y = img_H_tensor.to(device)
        
            # Resolution degradation
            H_funcs = SuperResolution(n_channels, img_size, sf, device)
            
            y_0 = H_funcs.H(y)
            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
            
            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], n_channels, img_size, img_size)
            pinv_y_0 = pinv_y_0 / 2 + 0.5
            
            # --------------------------------
            # (3) main iterations
            # --------------------------------

            # create sequence of timestep for sampling
            if skip_type == 'uniform':
                seq = [i * skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps - 1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps ** 2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            
            x_rand = torch.randn(y_0.shape[0], n_channels, img_size, img_size, device=device)
        
            x_0, _ = generation_steps(x_rand, seq, model, betas, H_funcs, y_0, sigma_0, \
                etaB=etaB, etaA=eta, etaC=eta, cls_fn=None, classes=None)
            x_0 = x_0[-1].to(device)
            x_0 = x_0 / 2 + 0.5

            # --------------------------------
            # (4) Result processing
            # --------------------------------

            img_E = utils_function.tensor2uint(x_0)
            img_H = utils_function.tensor2uint(y / 2 + 0.5)
                
            psnr = utils_function.calculate_psnr(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            ssim = utils_function.calculate_ssim(img_E, img_H, border=border)
            test_results['ssim'].append(ssim)

            logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB, SSIM: {:.4f}'.format(idx + 1, img_name+ext, psnr, ssim))
            
            if save_E:
                utils_function.imsave(img_E, os.path.join(E_path, img_name + '_E' + ext))

            if save_L:
                utils_function.imsave(utils_function.tensor2uint(pinv_y_0), os.path.join(E_path, img_name + '_L' + ext))
        
        # --------------------------------
        # Average PSNR and SSIM
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}): {:.4f} dB'.format(testset_name, ave_psnr))

        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('------> Average SSIM of ({}): {:.4f}'.format(testset_name, ave_ssim))

    # experiments
    running()


if __name__ == '__main__':

    main()
