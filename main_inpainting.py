import os.path
import cv2
import logging

import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict

from utils import utils_model
from utils import utils_logger
from utils import utils_image as util

from utils.svd_replacement import Inpainting
from utils.denoising import efficient_generalized_steps

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

    noise_level_img         = 0.0/255.0           # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'diffusion_ffhq_10m'  # 256x256_diffusion_uncond, diffusion_ffhq_10m; set diffusino model
    testset_name            = 'demo_test'        # set testing set, 'imagenet_val' | 'ffhq_val'
    num_train_timesteps     = 1000
    iter_num                = 20              # set number of iterations
    skip                    = num_train_timesteps//iter_num     # skip interval

    show_img                = False             # default: False
    save_L                  = False             # save LR image
    save_E                  = False              # save estimated image
    save_LEH                = True             # save zoomed LR, E and H images
    save_progressive        = False             # save generation process
    save_progressive_mask   = False             # save generation process

    sigma_0                 = 0
    lambda_                 = 1.                # key parameter lambda
    zeta                    = 1.0                      
    
    generate_mode           = 'DDRM'           # DDRM
    skip_type               = 'quad'            # uniform, quad
    
    task_current            = 'ip'              # 'ip' for inpainting
    n_channels              = 3                 # fixed
    cwd                     = ''
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results')      # fixed
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_maks_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta_zeta{zeta}_lambda{lambda_}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    calc_LPIPS              = True
    
    # noise schedule 
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)

    ##############################
    # paths
    ##############################

    L_path                  = os.path.join(testsets, testset_name)      # L_path, for Low-quality images
    E_path                  = os.path.join(results, result_name)        # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name             = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger                  = logging.getLogger(logger_name)

    ##############################
    # load model
    ##############################

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('zeta:{:.3f}, lambda:{:.3f} '.format(zeta, lambda_))
    logger.info('skip_type:{}, skip interval:{}'.format(skip_type, skip))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    def test_rho(lambda_=lambda_,zeta=zeta):
        logger.info('zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{}'.format(zeta, lambda_, ''))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []

        for idx, img in enumerate(L_paths):

            ##############################
            # load img
            ##############################

            idx += 1
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)

            ##############################
            # mask
            ##############################
            # loaded = np.load("inp_masks/lolcat_extra.npy")
            loaded = np.load("inp_masks/lorem3.npy")
            mask_n = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask_n == 0).long().reshape(-1) * 3
            
            mask = loaded
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
            
            # missing_r = torch.randperm(img_H.shape[0]**2)[:img_H.shape[0]**2 // 2].to(device).long() * 3
            # mask = missing_r.cpu().numpy()
                
            img_L = img_H * mask  / 255.   #(256,256,3)         [0,1]

            np.random.seed(seed=0)  # for reproducibility
            img_L = img_L * 2 - 1
            img_L += np.random.normal(0, noise_level_img * 2, img_L.shape) # add AWGN
            img_L = img_L / 2 + 0.5
            img_L = img_L * mask
            
            y = util.single2tensor4(img_L).to(device)   #(1,3,256,256)
            y = y * 2 -1        # [-1,1]
            mask = util.single2tensor4(mask.astype(np.float32)).to(device) 

            y_0 = y
            
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)

            ##############################
            # main iterations
            ##############################

            progress_img = []
            # create sequence of timestep for sampling
            if skip_type == 'uniform':
                seq = [i*skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps-1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::(len(seq)//10)]
            progress_seq.append(seq[-1])

            ###################################################
            H_funcs = Inpainting(n_channels, img_L.shape[0], missing, device)

            y_0 = H_funcs.H(y_0)
            x = torch.randn(*y.shape, device=device)      
            x_0, _ = efficient_generalized_steps(x, seq, model, betas, H_funcs, y_0, \
                                                         sigma_0, etaB=0.85, etaA=0.95, etaC=0.85, cls_fn=None, classes=None)
            x_0 = x_0[-1].to(device)
            
            ##############################
            # save
            ##############################
            x_0 = (x_0/2+0.5)

            # recover conditional part
            if generate_mode in ['repaint','DDRM']:
                x[mask.to(torch.bool)] = y[mask.to(torch.bool)]
            
            img_E = util.tensor2uint(x_0)
                
            psnr = util.calculate_psnr(img_E, img_H, border=0)  # change with your own border
            test_results['psnr'].append(psnr)
                    
            if calc_LPIPS:
                img_H_tensor = np.transpose(img_H, (2, 0, 1))
                img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                img_H_tensor = img_H_tensor / 255 * 2 -1
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score)
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(idx, img_name+ext, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
            else:
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB'.format(idx, img_name+ext, psnr))
                pass

            if save_E:
              util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+ext))

            if save_L:
              util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_L'+ext))

            if save_LEH:
              util.imsave(np.concatenate([util.single2uint(img_L), img_E, img_H], axis=1), os.path.join(E_path, img_name+model_name+'_LEH'+ext))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                if generate_mode in ['repaint','DDRM']:
                    mask = np.squeeze(mask.cpu().numpy())
                    if mask.ndim == 3:
                        mask = np.transpose(mask, (1, 2, 0))
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total,figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+'_process_lambda_{:.3f}_{}{}'.format(lambda_,current_time,ext)))
                images = []
                y_t = np.squeeze((y/2+0.5).cpu().numpy())
                if y_t.ndim == 3:
                    y_t = np.transpose(y_t, (1, 2, 0))
                if generate_mode in ['repaint','DDRM']:
                    for x in progress_img:
                        images.append((y_t)* mask+ (1-mask) * x)
                    img_total = cv2.hconcat(images)
                    if show_img:
                        util.imshow(img_total,figsize=(80,4))
                    if save_progressive_mask:
                        util.imsave(img_total*255., os.path.join(E_path, img_name+'_process_mask_lambda_{:.3f}_{}{}'.format(lambda_,current_time,ext)))

        ##################################
        # LPIPS and PSNR
        ##################################

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}): {:.4f} dB'.format(testset_name, noise_level_model, ave_psnr))

        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}), sigma: ({:.3f}): {:.4f}'.format(testset_name, noise_level_model, ave_lpips))

    # experiments
    lambdas = [lambda_*i for i in range(1,2)]
    for lambda_ in lambdas:
        for zeta_i in [zeta*i for i in range(1,2)]:
            test_rho(lambda_, zeta=zeta_i)

if __name__ == '__main__':
    main()
