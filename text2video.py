import os

import imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, TextToVideoZeroPipeline

OUTPUT_DIR = "test_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def text2video_zero(prompt, outfile: str | None = None):
    if outfile is None:
        sanitized_prompt = prompt.replace(" ", "_")
        model_dir = "text2video_zero"
        outfile = os.path.join(OUTPUT_DIR, model_dir, f"{sanitized_prompt}.mp4")

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = TextToVideoZeroPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda")

    result = pipe(prompt=prompt).images
    result = [(r * 255).astype("uint8") for r in result]
    imageio.mimsave(outfile, result, fps=4)


def text2video_zero_long(
    prompt,
    outfile: str | None = None,
    seed: int = 42,
    video_length: int = 128,
    chunk_size: int = 32,
):
    if outfile is None:
        sanitized_prompt = prompt.replace(" ", "_")
        model_dir = "text2video_zero"
        outfile = os.path.join(OUTPUT_DIR, model_dir, f"{sanitized_prompt}_long.mp4")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = TextToVideoZeroPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Generate the video chunk-by-chunk
    result = []
    chunk_ids = np.arange(0, video_length, chunk_size - 1)
    generator = torch.Generator(device="cuda")
    for i in range(len(chunk_ids)):
        print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
        ch_start = chunk_ids[i]
        ch_end = video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
        # Attach the first frame for Cross Frame Attention
        frame_ids = [0] + list(range(ch_start, ch_end))
        # Fix the seed for the temporal consistency
        generator.manual_seed(seed)
        output = pipe(
            prompt=prompt,
            video_length=len(frame_ids),
            generator=generator,
            frame_ids=frame_ids,
        )
        result.append(output.images[1:])

    # Concatenate chunks and save
    result = np.concatenate(result)
    result = [(r * 255).astype("uint8") for r in result]
    imageio.mimsave(outfile, result, fps=4)


def zeroscope(
    prompt: str,
    outfile: str | None = None,
    num_inference_steps: int = 40,
    num_frames: int = 24,
    height: int = 320,
    width: int = 576,
):
    if outfile is None:
        sanitized_prompt = prompt.replace(" ", "_")
        model_dir = "text2video_zero"
        outfile = os.path.join(OUTPUT_DIR, model_dir, f"{sanitized_prompt}.mp4")
    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    video_frames = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
    ).frames

    result = [(r).astype("uint8") for r in video_frames]
    imageio.mimsave(outfile, result, fps=4)
    output_gif_filepath = outfile.replace(".mp4", ".gif")
    # Make gif
    from utils import vid_to_gif

    vid_to_gif(outfile, output_gif_filepath, size=256)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Darth Vader is surfing on waves")
    parser.add_argument("--model_type", type=str, default="text2video_zero")
    args = parser.parse_args()

    if args.model_type == "text2video_zero":
        text2video_zero(args.prompt)
    elif args.model_type == "text2video_zero_long":
        text2video_zero_long(args.prompt)
    elif args.model_type == "zeroscope":
        zeroscope(args.prompt)
