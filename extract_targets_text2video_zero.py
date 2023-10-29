'''Script to extract conditioning vectors and latent vectors from a zeroscope video to video pipeline'''

import argparse
import os
import torch
from diffusers import TextToVideoZeroPipeline
from einops import rearrange
from PIL import Image
import numpy as np
import json
from tqdm import trange, tqdm
from utils import save_vectors_npy

MODEL_TYPE = "text2video_zero"


def main(args):
    # Load stuff needed for zeroscope
    pipe = TextToVideoZeroPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )
    pipe.to(args.gpu)

    # Load videos as tensors with shape (b f c h w)
    # We assume that we load BOLDMoments data, which has 45 frames per video and width and height of 268

    print("Getting z targets")
    # Get z
    get_and_save_z_targets(
        pipe,
        args.path_to_video_frames,
        batch_size=8,
        output_path=os.path.join(args.output_path, f'z_{MODEL_TYPE}_unflattened'),
    )

    print("Getting c targets")
    # Get c
    get_and_save_c_targets(
        pipe,
        args.path_to_annots,
        batch_size=None,
        output_path=os.path.join(args.output_path, f'c_{MODEL_TYPE}_unflattened'),
    )

    print(f"Saved target vectors to {args.output_path}")


def get_and_save_z_targets(
    pipe,
    path_to_video_frames,
    batch_size=None,
    output_path=f'./data/target_vectors/z_{MODEL_TYPE}',
):
    n_frames_to_load = 15
    max_frame_number = 45
    skip_frames = max_frame_number // n_frames_to_load
    print(f"Loading {n_frames_to_load} frames per video")
    size = 268
    n_videos = len(os.listdir(path_to_video_frames))
    video_folders = sorted(os.listdir(path_to_video_frames))

    if batch_size is None:
        batch_size = n_videos  # Requires 82GB RAM to operate with the full array

    for b in tqdm(
        range(n_videos // batch_size + 1),
        desc="Extracting and saving z vectors...",
        unit="batch",
    ):
        # Define varying batch size to catch the last batch
        bs = min(batch_size, n_videos - b * batch_size)

        # Build empty array to store videos
        videos = np.zeros((bs, n_frames_to_load, 3, size, size))

        # Load videos frame by frame
        for v, frame_folder in enumerate(
            video_folders[b * batch_size : (b + 1) * batch_size]
        ):
            for f in range(n_frames_to_load):
                # Try to load frame. If frame doesn't exist, use the last frame that was loadable
                try:
                    frame_path = os.path.join(
                        path_to_video_frames, frame_folder, f'{f*skip_frames+1:03d}.png'
                    )
                    frame = Image.open(frame_path).convert('RGB')
                except Exception:
                    frame = last_frame  # First frame should never fail, so last_frame should always have a value at this point
                    print(
                        f"Failed to load frame {f*skip_frames+1:03d} for video {frame_folder}. Using last frame instead."
                    )

                # Normalize frame
                norm_frame = (np.array(frame).transpose(2, 0, 1) / 255.0) * 2.0 - 1.0

                # Store in videos array with shape (b f c h w)
                videos[v, f] = norm_frame

        # Get latent embedding targets
        batch_of_zs = videos_to_latent_vectors(videos, pipe, return_flattened=False)

        print("batch_of_zs.shape", batch_of_zs.shape)
        # Save targets
        video_names = video_folders[b * batch_size : (b + 1) * batch_size]
        save_vectors_npy(batch_of_zs, output_path, video_names)


def get_and_save_c_targets(
    pipe,
    path_to_annots='./data/annotations.json',
    batch_size=None,
    output_path='./data/target_vectors/c_zeroscope',
    average_captions=True,
):
    # Load annotations (default: BOLDMoments)
    annots = json.load(
        open(path_to_annots, 'r')
    )  # captions located in annots.values()[0]['text_descriptions']

    # Get conditioning vectors
    for video_name, a in tqdm(
        annots.items(), desc="Extracting conditioning vectors...", unit="video"
    ):
        conditioning_vectors = prompts_to_conditioning_vectors(
            a['text_descriptions'], pipe
        )
        # average conditioning vectors
        if average_captions:
            c = torch.mean(conditioning_vectors, dim=0, keepdim=True)
        else:  # Choose first caption
            c = conditioning_vectors[0].unsqueeze(0)

        # save individual npy file
        save_vectors_npy(c.cpu().numpy(), output_path, [video_name])


def images_to_latent_vectors():
    pass


def load_nsd_image():
    img = nsda.read_images(s - imgidx[0])
    if plot:
        from matplotlib import pyplot as plt

        plt.imshow(img)
        plt.title(p['caption'])
        plt.show()
    init_image = load_img_from_arr(img, resolution).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

    return init_image


def videos_to_latent_vectors(pixels, pipe, batch_size: int = 60, return_flattened=True):
    nf = pixels.shape[1]
    nv = pixels.shape[0]
    pixels = rearrange(pixels, "v f c h w -> (v f) c h w")

    latents = []
    for idx in trange(
        0,
        pixels.shape[0],
        batch_size,
        desc="Encoding to latents...",
        unit_scale=batch_size,
        unit="frame",
    ):
        pixels_batch = torch.tensor(pixels[idx : idx + batch_size]).to(
            pipe.device, dtype=torch.half
        )
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

    print("latents.shape", latents.shape)
    return latents


def prompts_to_conditioning_vectors(prompts, pipe, return_flattened=False):
    with torch.no_grad():
        cond_vectors = pipe._encode_prompt(
            prompts,
            "cuda:0",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    if return_flattened:
        return rearrange(cond_vectors, "b k l -> b (k l)")
    return cond_vectors


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        required=False,
        type=str,
        default='stabilityai/stable-diffusion-2',
        help="Path to the videos for which to extract target vectors",
    )

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
        default='./data/stimuli/frames',
        help="Path to the videos for which to extract target vectors",
    )

    parser.add_argument(
        "--path_to_annots",
        required=False,
        type=str,
        default='./data/annotations.json',
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
