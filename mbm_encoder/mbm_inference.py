import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

import tqdm
from dataset import bmd_dataset
from sc_mbm.mae_for_fmri import MAEforFMRI
import argparse
import copy

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sd = torch.load(args.pretrain_model, map_location='cpu')
    config = sd['config']

    model = MAEforFMRI(num_voxels=config.num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
                num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                focus_range=config.focus_range, focus_rate=config.focus_rate, 
                img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss)

    print('Loading checkpoint from %s'%args.pretrain_model)
    model.load_state_dict(sd['model'], strict=True)
    model.to(device)
    
    print(config.subjects)

    dataset_test = bmd_dataset(dataset_path=args.dataset_path, roi_list=args.roi_list, patch_size=config.patch_size,
        transform=fmri_transform, aug_times=config.aug_times, num_sub_limit=config.num_sub_limit, subjects=config.subjects, split='test')

    print(f'Dataset size: {len(dataset_test)}\nNumber of voxels: {dataset_test.num_voxels}')
    dataloader_test = DataLoader(dataset_test, batch_size=1, sampler=None, 
                shuffle=False, pin_memory=True)

    total_loss = 0
    for fmri_data in tqdm.tqdm(dataloader_test):
        # compute loss
        loss, _, _ = model(fmri_data['fmri'].to(device), None, valid_idx=None, mask_ratio=config.mask_ratio)
        total_loss += loss.item()

        # extract features
        latent, _, _ = model.forward_encoder(fmri_data['fmri'].to(device), 0.0)

    total_loss /= len(dataloader_test)
    print("Avg loss on test set: {:.3f}".format(total_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MBM inference for fMRI', add_help=False)
    parser.add_argument('--pretrain_model', type=str)
    parser.add_argument('--roi_list', type=str, default='./roi_list/roi_list_reduced23.txt')
    parser.add_argument('--dataset_path', type=str, default='/gpfs/u/home/SIFA/SIFApnbw/scratch/mind-vis/data/bmd')

    args = parser.parse_args()
    main(args)