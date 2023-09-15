
# Import 
import os
import numpy
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import shutil


def main(args):

    pipe = DiffusionPipeline.from_pretrained("../zeroscope_v2_576w", 
                                             torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')

    video_frames = pipe(args.prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    video_path = export_to_video(video_frames)

    # Move video to output folder
    output_path = f'../zeroscope_test_generations/{args.prompt}.mp4'
    output_gif_filepath = f'../zeroscope_test_generations/{args.prompt}.gif'
    shutil.move(video_path, output_path)
    print(output_path)

    # Make gif
    from utils import vid_to_gif
    vid_to_gif(output_path, output_gif_filepath, size=256)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Darth Vader is surfing on waves")
    args = parser.parse_args()

    main(args)