"""Script to reconstruct videos from predicted 
latent and conditioning vectors using Zeroscope"""


import os
import numpy as np
import torch
import argparse
from diffusers import VideoToVideoSDPipeline, DPMSolverMultistepScheduler
from einops import rearrange
from utils import vid_to_gif, frames_to_vid

def main():

    parser = argparse.ArgumentParser(description='Reconstruct videos from predicted latents and conditioning vectors')

    parser.add_argument('--z_path', 
                        type=str, 
                        help='Path to predicted latents npy file', 
                        default='./estimated_vectors/WB_z_zeroscope'
                        )

    parser.add_argument('--c_path',
                        type=str,
                        help='Path to predicted conditioning vectors npy file',
                        default='./estimated_vectors/WB_c_zeroscope'
                        )

    parser.add_argument('--output_path',
                        type=str,
                        help='Output path for reconstructed videos',
                        default='./reconstructions/whole_brain'
                        )       
    
    parser.add_argument('--set',
                        type=str,
                        help='Set to reconstruct videos from. Options: train, test',
                        default='test'
                        )

    parser.add_argument('--use_gt_vecs',
                        type=bool,
                        help='Whether to use ground truth conditioning vectors',
                        default=False
                        )

    args = parser.parse_args()

    # Follow vid2vid approach: https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py

    pipe = VideoToVideoSDPipeline.from_pretrained("../zeroscope_v2_576w", 
                                                torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')

    if args.set == 'train':
        filename = 'preds_train.npy'
    elif args.set == 'test':
        filename = 'preds_test.npy'
    
    # Load predicted latents
    z = np.load(os.path.join(args.z_path, filename))

    # Reshape latents to expected shape: (b c f w h) with c = 4, f = 15
    z = rearrange(z, 'b (f c w h) -> b c f w h', f=15, c=4, w=33, h=33)
    z = torch.tensor(z).float().cuda()

    print('z.shape',z.shape)

    # Load conditioning vectors
    c = np.load(os.path.join(args.c_path, filename))
    
    # Reshape cond vectors
    c = rearrange(c, 'b (k l) -> b k l', k=77)
    c = torch.tensor(c).float().cuda()
    print('c.shape', c.shape)

    for i in range(len(z)):
    
        # Reconstruct videos
        print("Reconstructing video", i)
        rec_vid_frames =  pipe(video = z[i], 
                        prompt_embeds = c[i].unsqueeze(0), 
                        strength = 0.3, # Strength controls the noise applied to the latent before starting the diffusion process. Higher strength = higher noise. 
                        ).frames
        
        print("rec_vid_frames len", len(rec_vid_frames), "rec_vid_frames[0] shape", rec_vid_frames[0].shape)
        # Save reconstructed videos as mp4 and gif
        vid_path = os.path.join(args.output_path, 'mp4')
        os.makedirs(vid_path, exist_ok=True)
        gif_path = os.path.join(args.output_path, 'gif')
        os.makedirs(gif_path, exist_ok=True)
        
        frames_to_vid(rec_vid_frames, os.path.join(vid_path, f'{i+1:04d}.mp4'), fps=15)
        vid_to_gif(os.path.join(vid_path, f'{i+1:04d}.mp4'), os.path.join(gif_path, f'{i+1:04d}.gif'), size=268)


if __name__ == '__main__':
    main()                





