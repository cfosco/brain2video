
# Import 
import os
import numpy
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video



def main(args):

    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", 
                                             torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    video_frames = pipe(args.prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    export_to_video(video_frames, args.output_path)

    print("Video generated and saved to", args.output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Darth Vader is surfing on waves")
    parser.add_argument("--output_path", type=str, default="output.mp4")
    args = parser.parse_args()

    main(args)