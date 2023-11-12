import os

import cv2
import imageio
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    TextToVideoZeroPipeline,
    UniPCMultistepScheduler,
)
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
    CrossFrameAttnProcessor,
)
from PIL import Image

OUTPUT_DIR = "test_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def text2video_zero(prompt, outfile: str | None = None):
    if outfile is None:
        sanitized_prompt = prompt.replace(" ", "_")
        model_dir = "text2video_zero"
        outfile = os.path.join(OUTPUT_DIR, model_dir, f"{sanitized_prompt}.mp4")

    model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "stabilityai/stable-diffusion-2"
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
    # model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "stabilityai/stable-diffusion-2"
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


def canny_edge(image: Image.Image):
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def make_controlnet_pipeline(variant: str) -> StableDiffusionControlNetPipeline:
    assert variant in ["openpose", "edge", "canny", "depth"]
    controlnet = ControlNetModel.from_pretrained(
        f"lllyasviel/sd-controlnet-{variant}", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safey_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)  # type: ignore
    pipe.enable_model_cpu_offload()
    return pipe


def text2video_zero_posecontrol():
    pipe = make_controlnet_pipeline("openpose")
    from huggingface_hub import hf_hub_download

    filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
    repo_id = "PAIR/Text2Video-Zero"
    video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
    import shutil

    import imageio
    from PIL import Image

    shutil.copy(video_path, "input_video.mp4")

    reader = imageio.get_reader(video_path, "ffmpeg")
    frame_count = 16
    pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )

    model_id = "runwayml/stable-diffusion-v1-5"
    controlnet_id = "lllyasviel/sd-controlnet-openpose"
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    # Set the attention processor
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

    # fix latents for all frames
    latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(
        len(pose_images), 1, 1, 1
    )

    prompt = "Darth Vader dancing in a desert"
    out = []
    for inds in chunk(range(len(pose_images)), 8):
        pimage = [pose_images[i] for i in inds]
        batch_latents = latents[inds]
        result = pipe(
            prompt=[prompt] * len(pimage), image=pimage, latents=batch_latents
        ).images
        out.append(result)
    result = np.concatenate(out)
    imageio.mimsave("video.mp4", result, fps=4)


def chunk(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def _text2video_zero_depth():
    # depth_estimator = pipeline('depth-estimation')
    seed = 42
    video_path = "data/10_videos_for_reconstruction_test/mp4/0045.mp4"
    reader = imageio.get_reader(video_path, "ffmpeg")
    from extract_targets_depth import load_midas

    midas, transform, device = load_midas()
    depth_images = []
    for image in reader:
        with torch.no_grad():
            image = transform(image).to(device)
            depth = midas(image)
            depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())
            depth = (
                depth.squeeze()[..., None]
                .repeat(1, 1, 3)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

        depth_image = Image.fromarray(depth)
        depth_image = depth_image.resize((512, 512))
        depth_images.append(depth_image)

    depth_images = depth_images[::4]

    imageio.mimsave("depth_input_video.mp4", depth_images, fps=4)
    pipe = make_controlnet_pipeline("depth")

    # Set the attention processor
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

    # fix latents for all frames
    latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(
        len(depth_images), 1, 1, 1
    )
    prompt = "a baby licking a spatula"
    out = []
    generator = torch.Generator(device="cuda")
    for inds in chunk(range(len(depth_images)), 12):
        dimage = [depth_images[i] for i in inds]
        batch_latents = latents[inds]
        generator.manual_seed(seed)
        result = pipe(
            prompt=[prompt] * len(dimage),
            image=dimage,
            latents=batch_latents,
            generator=generator,
            video_length=len(dimage),
        ).images
        out.append(result)
    result = np.concatenate(out)
    imageio.mimsave("depth_output_video.mp4", result, fps=6)


def text2video_zero_depth():

    seed = 42
    video_path = "data/10_videos_for_reconstruction_test/mp4/0045.mp4"
    reader = imageio.get_reader(video_path, "ffmpeg")
    from extract_targets_depth import load_midas

    midas, transform, device = load_midas()
    depth_images = []
    for image in reader:
        with torch.no_grad():
            image = transform(image).to(device)
            depth = midas(image)
            depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())
            depth = (
                depth.squeeze()[..., None]
                .repeat(1, 1, 3)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

        depth_image = Image.fromarray(depth)
        depth_image = depth_image.resize((512, 512))
        depth_images.append(depth_image)

    depth_images = depth_images[::4]

    imageio.mimsave("depth_input_video.mp4", depth_images, fps=4)
    pipe = make_controlnet_pipeline("depth")

    # Set the attention processor
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

    # fix latents for all frames
    latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(
        len(depth_images), 1, 1, 1
    )
    prompt = "a baby licking a spatula"
    out = []
    generator = torch.Generator(device="cuda")
    # use the same latents for all frames?
    batch_latents = latents[:12]
    for inds in chunk(range(len(depth_images)), 12):
        dimage = [depth_images[i] for i in inds]
        batch_latents = batch_latents[: len(dimage)]
        generator.manual_seed(seed)
        result = pipe(
            prompt=[prompt] * len(dimage),
            image=dimage,
            latents=batch_latents,
            generator=generator,
        ).images[1:]
        out.append(result)
    result = np.concatenate(out)
    imageio.mimsave("depth_output_video.mp4", result, fps=6)


def text2video_zero_depth_reconstruct():
    input_vector_path = 'data/target_vectors/depth_unflattened/0002.npy'
    input_vector = np.load(input_vector_path)
    print(input_vector.shape)
    image = input_vector[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    print(image.shape)
    print(image.min(), image.max())
    depth_image = Image.fromarray(image)
    # upscale image to 256x256
    depth_image = depth_image.resize((256, 256))
    depth_images = [depth_image] * 8
    imageio.mimsave("depth_reconstruct_input_video.mp4", depth_images, fps=4)

    pipe = make_controlnet_pipeline("depth")

    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

    with torch.inference_mode():
        prompt = "a smiling baby"
        out = pipe(
            prompt=[prompt] * len(depth_images),
            image=depth_images,
        ).images
        imageio.mimsave("depth_reconstruct_video.mp4", out, fps=4)


def text2video_zero_edge_control():
    from huggingface_hub import hf_hub_download

    filename = "__assets__/pix2pix video/camel.mp4"
    repo_id = "PAIR/Text2Video-Zero"
    video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

    import shutil

    import imageio
    from PIL import Image

    shutil.copy(video_path, "edge_control_input_video.mp4")

    reader = imageio.get_reader(video_path, "ffmpeg")
    frame_count = 8
    video = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

    prompt = "make it Van Gogh Starry Night style"
    result = pipe(prompt=[prompt] * len(video), image=video).images
    imageio.mimsave("edited_video.mp4", result, fps=4)


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

    # if args.model_type == "text2video_zero":
    #     text2video_zero(args.prompt)
    # elif args.model_type == "text2video_zero_long":
    #     text2video_zero_long(args.prompt)
    # elif args.model_type == "zeroscope":
    #     zeroscope(args.prompt)

    # text2video_zero_posecontrol()
    # text2video_zero_edge_control()
    text2video_zero_depth()
