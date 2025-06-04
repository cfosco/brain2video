'''Extract BLIP embeddings for BOLDMoments videos'''

import argparse
import os
import torch
from tqdm import tqdm
import sys
sys.path.append("./blip/models")
from blip.models.blip import blip_decoder
from utils import save_vectors_npy
from PIL import Image
import numpy as np
from torchvision import transforms
from einops import rearrange

from torch.utils.data import DataLoader

from dataset import *
from dataset import STIM_DATASET_MAP, DATASET_PATHS


# TODO: Refactor this repeating code from all the extract_ scripts
def main(args):

    res = 240

    # Build dataset and dataloader
    dataset = STIM_DATASET_MAP[args.dataset](
                args.input_path if args.input_path is not None else DATASET_PATHS[args.dataset]['stimuli'], 
                args.metadata_path if args.metadata_path is not None else DATASET_PATHS[args.dataset]['metadata'],
                subset='all',
                resolution=res,
                transform=None,
                normalize=True,
                return_filename=True,
                load_from_frames=args.load_from_frames,
                skip_frames=None,
                n_frames_per_video=15,
                minus_1_to_1=False) # BLIP needs tensor values to be between 0 and 1

    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) # will return batches as pytorch tensors
        
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Instantiate pipeline
    blip_pipeline = BLIPEmbeddingPipeline(device=device,
                                            return_flattened=False,
                                            res=res)

    # Define save path
    if args.save_path is None:
        args.save_path = f'./data/target_vectors_{args.dataset}/blip'

    # Extract and save z targets
    for batch, filenames in tqdm(dataloader, desc=f"Extracting BLIP visual vectors for {args.dataset}...", unit="batch"):    

        batch_of_blips = blip_pipeline(batch) 
        
        # Save targets
        save_vectors_npy(batch_of_blips, args.save_path, filenames)   
        




    ### --- OLD CODE

    # Load BLIP model
    # image_size = 240
    # model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    # device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # model = blip_decoder(pretrained=model_url, image_size=image_size, vit="base")
    # model.eval()
    # model = model.to(device)


    # Make feature
    # nsda = NSDAccess('../../../nsd/')
    # for s in tqdm(range(nimage)):
    #     img_arr = nsda.read_images(s)
    #     image = Image.fromarray(img_arr).convert("RGB").resize((image_size,image_size), resample=Image.LANCZOS)
    #     img_arr = transforms.ToTensor()(image).to('cuda').unsqueeze(0)
    #     with torch.no_grad():
    #         vit_feat = model.visual_encoder(img_arr).cpu().detach().numpy().squeeze()        
    #     np.save(f'{savedir}/{s:06}.npy',vit_feat)


    # get_and_save_BLIP_targets(model, 
    #                             args.path_to_video_frames, 
    #                             batch_size=8, 
    #                             size=image_size,
    #                             output_path=os.path.join(args.output_path, 'blip_unflattened'))


class BLIPEmbeddingPipeline():
    def __init__(self, device="cuda", return_flattened=False, res=240):
        model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
        self.model = blip_decoder(pretrained=model_url, image_size=res, vit="base")
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.return_flattened = return_flattened

    def __call__(self, video_batch):
        # Make into pytorch tensor if necessary and send to device
        if not isinstance(video_batch, torch.Tensor):
            video_batch = torch.tensor(video_batch)
        video_batch = video_batch.float().to(self.device)        

        nb = video_batch.shape[0]
        nf = video_batch.shape[1]
        pixels_batch = rearrange(video_batch, "b f c h w -> (b f) c h w")
        
        with torch.no_grad():
            emb_batch = self.model.visual_encoder(pixels_batch)

        emb_batch = rearrange(emb_batch, "(b f) p c -> b f p c", b=nb, f=nf)

        # average over frames
        emb_batch = torch.mean(emb_batch, dim=1)


        if self.return_flattened: # Shape of latents: ((v 15) 4 33 33) -> (v 65340)
            emb_batch = rearrange(emb_batch, "b p c -> b (p c)")

        return emb_batch.detach().cpu()



#### ---- DEPRECATED FUNCTIONS
    


def get_and_save_BLIP_targets_old(model, 
                              path_to_video_frames, 
                              batch_size=None, 
                              size=264,
                              output_path='./data/target_vectors/blip',
                              flatten=False):

    n_frames_to_load = 15
    max_frame_number = 45
    skip_frames = max_frame_number // n_frames_to_load
    print(f"Loading {n_frames_to_load} frames per video")
    n_videos = len(os.listdir(path_to_video_frames))
    video_folders = sorted(os.listdir(path_to_video_frames))
    
    if batch_size is None:
        batch_size = n_videos

    for b in tqdm(range(n_videos//batch_size+1), desc="Extracting and saving blip vectors...", unit="batch"):
        # Define varying batch size to catch the last batch
        bs = min(batch_size, n_videos-b*batch_size)

        batch_of_blip_embs = []        

        # Load videos frame by frame
        for v, frame_folder in enumerate(video_folders[b*batch_size:(b+1)*batch_size]):
            frames = []
            for f in range(n_frames_to_load):
                # Try to load frame. If frame doesn't exist, use the last frame that was loadable
                try:
                    frame_path = os.path.join(path_to_video_frames, frame_folder, f'{f*skip_frames+1:03d}.png')
                    frame = Image.open(frame_path).convert('RGB').resize((size,size), resample=Image.LANCZOS)
                except:
                    frame = last_frame # First frame should never fail, so last_frame should always have a value at this point
                    print(f"Failed to load frame {f*skip_frames+1:03d} for video {frame_folder}. Using last frame instead.")
                last_frame = frame

                frame = transforms.ToTensor()(frame).to('cuda')
                frames.append(frame)
            
            frames = torch.stack(frames, dim=0)
            
            # Get latent embedding targets
            with torch.no_grad():
                blip_emb = model.visual_encoder(frames).cpu().detach().numpy().squeeze()
            # print("blip_emb.shape", blip_emb.shape)
            # blip_emb.shape (15, 226, 768)

            # Average over frames
            blip_emb = np.mean(blip_emb, axis=0)

            # Shape of blip_emb: ((226, 768) -> (173568))
            
            if flatten:
                blip_emb = blip_emb.reshape(-1)
            batch_of_blip_embs.append(blip_emb)

           
        # Save targets
        video_names = video_folders[b*batch_size:(b+1)*batch_size]
        save_vectors_npy(batch_of_blip_embs, output_path, video_names)



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

