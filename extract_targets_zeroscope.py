'''Script to extract conditioning vectors and latent vectors from a zeroscope video to video pipeline'''

import argparse
import os
import pickle as pkl
import torch
from diffusers import DiffusionPipeline
from einops import rearrange
from PIL import Image
import numpy as np
import json
from tqdm import trange, tqdm
from utils import save_vectors_npy
from torch.utils.data import DataLoader
from dataset import NSDImageDataset

def main(args):

    ## Load stuff needed for zeroscope
    pipe = DiffusionPipeline.from_pretrained("../zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.to(args.gpu)

    # Print config of pipe
    print(pipe.config)
    
    # Print config of text encoder
    print(pipe.text_encoder.config)

    ## Load videos as tensors with shape (b f c h w)
    # We assume that we load BOLDMoments data, which has 45 frames per video and width and height of 268
    
    # print("Getting z targets")
    ## Get z
    # get_and_save_z_targets(pipe, 
    #                        args.path_to_video_frames, 
    #                        batch_size=8, 
    #                        output_path=os.path.join(args.output_path, 'z_zeroscope_unflattened'))



    # get_and_save_z_targets_nsd(pipe, 
    #                        args.nsd_path, 
    #                        batch_size=80, 
    #                        resolution=268, 
    #                        output_path=os.path.join(args.output_path, 'z_zeroscope'))

    # print("Getting c targets")
    ## Get c 
    get_and_save_c_targets(pipe, 
                           args.path_to_annots, 
                           batch_size=None, 
                           output_path=os.path.join(args.output_path, 'c_zeroscope'))

    # print(f"Saved target vectors to {args.output_path}")


def get_and_save_z_targets_had():

    dataset = HADVideoDataset(had_path='./data/stimuli_had',
                              resolution=268,
                              subset='train',
                              return_names=True)

    # Iterate over dataset and extract vectors
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)



def get_and_save_z_targets_nsd(pipe, 
                               nsd_path = '../StableDiffusionReconstruction/nsd',
                               idxs = list(range(0,73000)),
                               batch_size=None,
                               resolution=268,
                               output_path='./data/target_vectors_nsd/z_zeroscope',):
    
    # Instantiate Dataset
    dataset = NSDImageDataset(idxs = idxs, 
                              nsd_path=nsd_path,
                              resolution=resolution)

    names = [f'{idx:06d}' for idx in idxs]

    # Iterate over dataset and extract vectors
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for i, batch in enumerate(tqdm(dataloader, desc="Extracting and saving z vectors...", unit="batch")):

        # Get latent embedding targets
        batch_of_zs = images_to_latent_vectors(batch, pipe, return_flattened=False)

        # print("batch_of_zs.shape", batch_of_zs.shape)

        # Save targets
        save_vectors_npy(batch_of_zs, output_path, names[i*batch_size:(i+1)*batch_size])
    
    

def get_and_save_z_targets(pipe, path_to_video_frames, batch_size=None, output_path='./data/target_vectors/z_zeroscope'):

    n_frames_to_load = 15
    max_frame_number = 45
    skip_frames = max_frame_number // n_frames_to_load
    print(f"Loading {n_frames_to_load} frames per video")
    size = 268
    n_videos = len(os.listdir(path_to_video_frames))
    video_folders = sorted(os.listdir(path_to_video_frames))
    
    if batch_size is None:
        batch_size = n_videos # Requires 82GB RAM to operate with the full array

    for b in tqdm(range(n_videos//batch_size+1), desc="Extracting and saving z vectors...", unit="batch"):
        # Define varying batch size to catch the last batch
        bs = min(batch_size, n_videos-b*batch_size)

        # Build empty array to store videos
        videos = np.zeros((bs, n_frames_to_load, 3, size, size))

        ## TODO: Encapsulate the entire following loop in a dataloader-dataset combo

        # Load videos frame by frame
        for v, frame_folder in enumerate(video_folders[b*batch_size:(b+1)*batch_size]):
            for f in range(n_frames_to_load):
                # Try to load frame. If frame doesn't exist, use the last frame that was loadable
                try:
                    frame_path = os.path.join(path_to_video_frames, frame_folder, f'{f*skip_frames+1:03d}.png')
                    frame = Image.open(frame_path).convert('RGB')
                except:
                    frame = last_frame # First frame should never fail, so last_frame should always have a value at this point
                    print(f"Failed to load frame {f*skip_frames+1:03d} for video {frame_folder}. Using last frame instead.")
                last_frame = frame

                # Normalize frame 
                norm_frame = (np.array(frame).transpose(2, 0, 1) / 255.0 ) * 2.0 - 1.0

                # Store in videos array with shape (b f c h w)
                videos[v, f] = norm_frame
    
        # Get latent embedding targets
        batch_of_zs = videos_to_latent_vectors(videos, pipe, return_flattened=False)

        # print("batch_of_zs.shape", batch_of_zs.shape)
        # input()
        # Save targets
        video_names = video_folders[b*batch_size:(b+1)*batch_size]
        save_vectors_npy(batch_of_zs, output_path, video_names)


def get_and_save_c_targets(pipe, 
                           path_to_annots='./data/metadata_bmd/annotations.json', 
                           batch_size=None, 
                           output_path='./data/target_vectors_bmd/c_zeroscope',
                           average_captions=True):

    # Load annotations (default: BOLDMoments)
    annots = json.load(open(path_to_annots, 'r')) # captions located in annots.values()[0]['text_descriptions']
    
    # Get conditioning vectors
    for video_name, a in tqdm(annots.items(), desc="Extracting conditioning vectors...", unit="video"):
        conditioning_vectors = prompts_to_conditioning_vectors(a['text_descriptions'], pipe)
        # print("conditioning_vectors.shape", conditioning_vectors.shape) # [5, 77, 1024]

        # average conditioning vectors
        if average_captions:
            c = torch.mean(conditioning_vectors, dim=0, keepdim=True)
        else: # Choose first caption
            c = conditioning_vectors[0].unsqueeze(0) 


        print("c.shape", c.shape)

        # save individual npy file
        # save_vectors_npy(c.cpu().numpy(), output_path, [video_name])

def prompts_to_conditioning_vectors(prompts, pipe, return_flattened=False):
    with torch.no_grad():
        # Encode prompt returns the hidden_layer of the text encoder
        cond_vectors = pipe._encode_prompt(
                prompts,
                "cuda:0",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        
        print("cond_vectors.shape", cond_vectors.shape)
    if return_flattened:
        return rearrange(cond_vectors, "b k l -> b (k l)")
    return cond_vectors

def images_to_latent_vectors(pixels, pipe, return_flattened=False):

    pixels_batch = pixels.to(pipe.device, dtype=torch.half) # Expected to be of dim (b c h w)
    latents = []
    with torch.no_grad():
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
  
    lb = latents_batch.mul(pipe.vae.config.scaling_factor).detach().cpu()
    latents.append(lb)
    latents = torch.cat(latents)
    
    # Shape of latents: ((v 15) 4 33 33) -> (v 65340)
    if return_flattened:
        latents = rearrange(latents, "b c h w -> b (c h w)")

    return latents
    

def videos_to_latent_vectors(pixels, pipe, batch_size: int = 60, return_flattened=False):
    nf = pixels.shape[1]
    nv = pixels.shape[0]
    pixels = rearrange(pixels, "v f c h w -> (v f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", 
        unit_scale=batch_size, unit="frame"
    ):  
        # TODO CHECK IF DIMENSIONALITY MATCHES WHAT ZEROSCOPE'S vae.encode EXPECTS
        pixels_batch = torch.tensor(pixels[idx : idx + batch_size]).to(pipe.device, dtype=torch.half)
        # print("pixels_batch generated. Shape:", pixels_batch.shape)
        # print_current_gpu_memory()
        with torch.no_grad():
            latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        # print("latents_batch generated. Shape:", latents_batch.shape)
        # print_current_gpu_memory()
        lb = latents_batch.mul(pipe.vae.config.scaling_factor).detach().cpu()
        latents.append(lb)
    latents = torch.cat(latents)

    # Shape of latents: ((v 15) 4 33 33) -> (v 65340)

    if return_flattened:
        latents = rearrange(latents, "(v f) c h w -> v (f c h w)", v=nv, f=nf)
    else:
        latents = rearrange(latents, "(v f) c h w -> v f c h w", v=nv, f=nf)
    
    # print("latents.shape", latents.shape)
    return latents


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extract_for_idx",
        required=False,
        default=None,
        nargs="*",
        type=int,
        help="Start and end idx of the videos for which to extract target vectors",
    )

    parser.add_argument(
        "--path_to_video_frames",
        required=False,
        type=str,
        default='./data/stimuli_bmd/frames',
        help="Path to the videos for which to extract target vectors",
    )

    parser.add_argument(
        "--nsd_path",
        required=False,
        type=str,
        default='../StableDiffusionReconstruction/nsd',
        help="Path to the NSD dataset",
    )

    parser.add_argument(
        "--path_to_annots",
        required=False,
        type=str,
        default='./data/metadata_bmd/annotations.json',
        help="Path to the BOLDMoments annotations file",
    )

    parser.add_argument(
        "--output_path",
        required=False,
        type=str,
        default='./data/target_vectors',
        help="Path to store the target vectors. Vectors will be stored in subfolders z_zeroscope and c_zeroscope",
    )

    parser.add_argument(
        "--gpu",
        required=False,
        type=str,
        default='cuda:0',
        help="GPU to use for extracting target vectors",
    )

    args = parser.parse_args()

    main(args)