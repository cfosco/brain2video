"""Extract BLIP embeddings for BOLDMoments videos"""
import argparse
import json
import os

import cv2
import ffmpeg
import imageio
import numpy as np
import torch
from tqdm import tqdm

from utils import save_vectors_npy


def get_num_frames(video_path):
    """Get number of frames in video"""
    probe = ffmpeg.probe(video_path)

    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    return int(video_stream["nb_frames"])


def load_midas(model_type="DPT_Hybrid"):
    # Load the midas depth estimator
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    print(f"Loaded model: {model_type} in {device}")

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform, device


def main(args):
    # load depth model
    midas, transform, device = load_midas()

    # Make feature
    # nsda = NSDAccess('../../../nsd/')
    # for s in tqdm(range(nimage)):
    #     img_arr = nsda.read_images(s)
    #     image = Image.fromarray(img_arr).convert("RGB").resize((image_size,image_size), resample=Image.LANCZOS)
    #     img_arr = transforms.ToTensor()(image).to('cuda').unsqueeze(0)
    #     with torch.no_grad():
    #         vit_feat = model.visual_encoder(img_arr).cpu().detach().numpy().squeeze()
    #     np.save(f'{savedir}/{s:06}.npy',vit_feat)

    # get_and_save_depth_targets(
    #     midas,
    #     transform,
    #     device,
    #     args.path_to_video_frames,
    #     batch_size=8,
    #     output_path=os.path.join(args.output_path, "depth_frames_unflattened"),
    # )

    data_root = "data/stimuli_had/"
    output_root = "data/target_vectors/had_test_set_depth_frames"
    metatdata_file = "./data/had_test_set_video_paths.json"
    # output_root = "data/target_vectors/had_train_set_depth_frames"
    # metatdata_file = "./data/had_train_set_video_paths.json"
    get_and_save_depth_targets_from_video(
        midas,
        transform,
        device,
        data_root,
        output_root,
        metadata_file=metatdata_file,
    )


def get_and_save_depth_targets_from_video(
    model,
    transform,
    device,
    data_root,
    output_root,
    metadata_file,
    num_frames: int = 8,
    downsample_factor: float = 6,
    frame_agg: str | None = None,
    batch_size=None,
    output_path="./data/target_vectors/depth",
    flatten=False,
):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    metadata = list(reversed(metadata))

    total = len(metadata)
    for path in tqdm(
        metadata, desc="Extracting and saving depth targets...", total=total
    ):
        full_path = os.path.join(data_root, path)
        reader = imageio.get_reader(full_path)
        num_frames = len(reader)
        num_frames = get_num_frames(full_path)
        # get 8 frames
        frame_inds = np.linspace(0, num_frames - 1, 8, dtype=int)
        frames = []
        for ind in frame_inds:
            frame = reader.get_data(ind)
            frame = transform(frame).to(device)
            frames.append(frame)

        frames = torch.cat(frames, dim=0).to(device)

        with torch.no_grad():
            prediction = model(frames)

        if frame_agg == "mean":
            # Average over frames
            # [num_frames, 384, 384] -> [384, 384]
            prediction = torch.mean(prediction, dim=0, keepdim=True)

        # downsample prediction by factor of downsample_factor
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(0),
            scale_factor=1 / downsample_factor,
            mode="bilinear",
            align_corners=False,
        )

        prediction = prediction.squeeze().cpu().numpy()  # [96, 96]
        prediction = (prediction - prediction.min()) / (
            prediction.max() - prediction.min()
        )
        prediction = (2 * prediction) - 1

        if flatten:
            prediction = prediction.reshape(-1)

        # Save targets

        save_path = os.path.join(output_root, path.replace(".mp4", ".npy"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, prediction)
        # Save each target vector for each video as its own npy file


def get_and_save_depth_targets(
    model,
    transform,
    device,
    path_to_video_frames,
    num_frames: int = 8,
    downsample_factor: float = 6,
    frame_agg: str | None = None,
    batch_size=None,
    output_path="./data/target_vectors/depth",
    flatten=False,
):
    max_frame_number = 45
    skip_frames = max_frame_number // num_frames
    print(f"Loading {num_frames} frames per video")
    n_videos = len(os.listdir(path_to_video_frames))
    video_folders = sorted(os.listdir(path_to_video_frames))

    if batch_size is None:
        batch_size = n_videos

    for b in tqdm(
        range(n_videos // batch_size + 1),
        desc="Extracting and saving depth targets...",
        unit="batch",
    ):
        # Define varying batch size to catch the last batch
        min(batch_size, n_videos - b * batch_size)

        batch_of_depth_targets = []

        # Load videos frame by frame
        for v, frame_folder in enumerate(
            video_folders[b * batch_size : (b + 1) * batch_size]
        ):
            frames = []
            for f in range(num_frames):
                # Try to load frame. If frame doesn't exist, use the last frame that was loadable
                try:
                    frame_path = os.path.join(
                        path_to_video_frames, frame_folder, f"{f*skip_frames+1:03d}.png"
                    )
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = transform(frame)
                except Exception as e:
                    print(e)
                    print(
                        "Frame",
                        f,
                        "of video",
                        frame_folder,
                        "not found. Using last frame that was loadable.",
                    )
                    break

                frames.append(frame)

            frames = torch.cat(frames, dim=0).to(device)

            with torch.no_grad():
                prediction = model(frames)

            # Average over frames
            # [num_frames, 384, 384] -> [384, 384]
            if frame_agg == "mean":
                prediction = torch.mean(prediction, dim=0, keepdim=True)

            # downsample prediction by factor of downsample_factor
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(0),
                scale_factor=1 / downsample_factor,
                mode="bilinear",
                align_corners=False,
            )

            prediction = prediction.squeeze().cpu().numpy()  # [96, 96]
            prediction = (prediction - prediction.min()) / (
                prediction.max() - prediction.min()
            )
            prediction = (2 * prediction) - 1

            if flatten:
                prediction = prediction.reshape(-1)
            batch_of_depth_targets.append(prediction)

        # Save targets
        video_names = video_folders[b * batch_size : (b + 1) * batch_size]
        save_vectors_npy(batch_of_depth_targets, output_path, video_names)


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
        "--output_path",
        required=False,
        type=str,
        default="./data/target_vectors",
        help="Path to store the target vectors. Vectors will be stored in subfolder blip",
    )

    parser.add_argument(
        "--path_to_video_frames",
        required=False,
        type=str,
        default="./data/stimuli/frames",
        help="Path to the videos for which to extract target vectors",
    )

    args = parser.parse_args()

    main(args)
