import torch
import os
import numpy as np
from einops import rearrange
from models.networks import NGP
import argparse
from utils import load_ckpt
import warnings; warnings.filterwarnings("ignore")
import scipy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='heart',
                        help='which dataset to train/test')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ne_fine = NGP().cuda()
    load_ckpt(ne_fine, f'./ckpts/{args.dataset_name}/{args.root_dir}/ckpt_slim.ckpt')
    @torch.no_grad()
    def ne_func(points):
        result = ne_fine(points)
        return result
    z,y,x = torch.meshgrid(torch.arange(160, dtype=torch.float32),
                        torch.arange(160, dtype=torch.float32),
                        torch.arange(160, dtype=torch.float32),indexing=None)
    dirs_x = x 
    dirs_y = y 
    dirs_z = z
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1).cuda()
    rays_dir = rays_dir/160 - 0.5
    rays_dir = rays_dir.reshape(-1, 3)
    res = ne_func(rays_dir)
    res = rearrange(res.cpu().numpy(), '(h w d) c -> h w d c', h=160,w=160)
    res = res.squeeze(3)
    print(res.shape)
    res = (res * 255).astype(np.uint8)
    filename = os.path.basename(args.root_dir)
    scipy.io.savemat(os.path.join('./',filename +'_3d.mat'),{'gray':res})