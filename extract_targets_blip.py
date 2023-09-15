'''Extract BLIP embeddings for BOLDMoments videos'''

import argparse
import os
import pickle as pkl
import torch
from tqdm import tqdm
import sys
sys.path.append("./blip/models")
from blip import blip_decoder
from utils import save_vectors_npy
from PIL import Image
import numpy as np
from torchvision import transforms

def main():

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
        "--output_path",
        required=False,
        type=str,
        default='./data/target_vectors',
        help="Path to store the target vectors. Vectors will be stored in subfolder blip",
    )

    parser.add_argument(
        "--path_to_video_frames",
        required=False,
        type=str,
        default='./data/stimuli/frames',
        help="Path to the videos for which to extract target vectors",
    )

    args = parser.parse_args()

    
    # Load BLIP model
    image_size = 240
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit="base")
    model.eval()
    model = model.to(device)


    # Make feature
    # nsda = NSDAccess('../../../nsd/')
    # for s in tqdm(range(nimage)):
    #     img_arr = nsda.read_images(s)
    #     image = Image.fromarray(img_arr).convert("RGB").resize((image_size,image_size), resample=Image.LANCZOS)
    #     img_arr = transforms.ToTensor()(image).to('cuda').unsqueeze(0)
    #     with torch.no_grad():
    #         vit_feat = model.visual_encoder(img_arr).cpu().detach().numpy().squeeze()        
    #     np.save(f'{savedir}/{s:06}.npy',vit_feat)


    get_and_save_BLIP_targets(model, 
                                args.path_to_video_frames, 
                                batch_size=8, 
                                size=image_size,
                                output_path=os.path.join(args.output_path, 'blip'))



def get_and_save_BLIP_targets(model, 
                              path_to_video_frames, 
                              batch_size=None, 
                              size=240,
                              output_path='./data/target_vectors/blip'):

    n_frames_to_load = 15
    max_frame_number = 45
    skip_frames = max_frame_number // n_frames_to_load
    print(f"Loading {n_frames_to_load} frames per video")
    size = 240
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
            blip_emb = blip_emb.reshape(-1)
            batch_of_blip_embs.append(blip_emb)

           
        # Save targets
        video_names = video_folders[b*batch_size:(b+1)*batch_size]
        save_vectors_npy(batch_of_blip_embs, output_path, video_names)



if __name__ == "__main__":
    main()


