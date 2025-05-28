'''File to extract regression targets from SVD's image encoder and text encoders'''

import argparse
import os
import torch
from einops import rearrange
from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import json
from tqdm import trange, tqdm
from utils import save_vectors_npy
from torch.utils.data import DataLoader
from dataset import NSDImageDataset
from dataset import CC2017VideoDataset


def main(args):  

    # Build dataset and dataloader
    dataset = DATASET_MAP[args.dataset](
                args.input_path if args.input_path is not None else DATASET_PATHS[args.dataset]['stimuli'], 
                args.metadata_path if args.metadata_path is not None else DATASET_PATHS[args.dataset]['metadata'],
                subset='all',
                resolution=268,
                transform=None,
                normalize=True,
                return_filename=True,
                load_from_frames=args.load_from_frames,
                skip_frames=None,
                n_frames_per_video=15)

    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) # will return batches as pytorch tensors
        
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Instantiate pipeline
    latent_pipeline = SVDLatentEmbeddingPipeline(device=device,
                                                return_flattened=False)

    # Define save path
    if args.save_path is None:
        args.save_path = f'./data/target_vectors_{args.dataset}/z_svd'

    # Extract and save z targets
    for batch, filenames in tqdm(dataloader, desc=f"Extracting z vectors for {args.dataset}...", unit="batch"):    

        # print("batch.shape", batch.shape)
        batch_of_zs = latent_pipeline(batch) 

        # print("names of vids in batch:",filenames)
        # print("batch_of_zs.shape", batch_of_zs.shape)
        
        # Save targets
        save_vectors_npy(batch_of_zs, args.save_path, filenames)   
        

class SVDLatentEmbeddingPipeline:
    def __init__(self, device="cuda", return_flattened=False):
        self.pipe = DiffusionPipeline.from_pretrained("../zeroscope_v2_576w", torch_dtype=torch.float16)
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        
        self.pipe = self.pipe.to(device)
        self.device = device
        self.return_flattened = return_flattened

    def __call__(self, video_batch):

        # Make into pytorch tensor if necessary and send to device
        if not isinstance(video_batch, torch.Tensor):
            video_batch = torch.tensor(video_batch)
        video_batch = video_batch.half().to(self.pipe.device)

        # nf = video_batch.shape[1]
        # nb = video_batch.shape[0]
        # pixels_batch = rearrange(video_batch, "b f c h w -> (b f) c h w")

        with torch.no_grad():
            image_latents = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        latents_batch = latents_batch.mul(self.pipe.vae.config.scaling_factor).detach().cpu()

        if self.return_flattened: # Shape of latents: ((v 15) 4 33 33) -> (v 65340)
            latents_batch = rearrange(latents_batch, "(b f) c h w -> b (f c h w)", b=nb, f=nf)
        else:
            latents_batch = rearrange(latents_batch, "(b f) c h w -> b f c h w", b=nb, f=nf)
        
        return latents_batch


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-idxs",
                        "--extract_for_idx",
                        required=False,
                        default=None,
                        nargs="*",
                        type=int,
                        help="Start and end idx of the videos for which to extract target vectors")

    parser.add_argument('-d', 
                        '--dataset',
                        type=str, 
                        required=True,
                        help='Dataset to extract embeddings from. One of bmd, bmd_captions, had, nsd, nod, cc2017')

    parser.add_argument('-i',
                        '--input_path', 
                        type=str, 
                        default=None, 
                        help='Path to the stimuli. Can be folder of frame_folder or folder of videos.')

    parser.add_argument('-m',
                        '--metadata_path', 
                        type=str, 
                        default=None, 
                        help='Path to the metadata.')

    parser.add_argument('-s',
                        '--save_path',
                        type=str, 
                        default=None, 
                        help='Path to save the extracted embeddings.')

    parser.add_argument('-f',
                        '--load_from_frames',
                        type=bool,
                        default=False,
                        help='Whether to load the video from frames (True) or from mp4 (False).')

    parser.add_argument('-dev',
                        '--device',
                        type=str,
                        default="cuda:0",
                        help='Device to run the model on.')
    
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=8,
                        help='Batch size for the dataloader.')
    
    args = parser.parse_args()

    main(args)