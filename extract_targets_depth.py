"""Extract BLIP embeddings for BOLDMoments videos"""

import argparse
import os

import cv2
import torch
from tqdm import tqdm

from utils import save_vectors_npy


def load_midas(model_type="DPT_Hybrid"):
    # Load the midas depth estimator
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    print(f"Loaded model: {model_type} in {device}")

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
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

    get_and_save_depth_targets(
        midas,
        transform,
        device,
        args.path_to_video_frames,
        batch_size=8,
        output_path=os.path.join(args.output_path, "depth_unflattened"),
    )


def get_and_save_depth_targets(
    model,
    transform,
    device,
    path_to_video_frames,
    batch_size=None,
    output_path="./data/target_vectors/depth",
    flatten=False,
):
    n_frames_to_load = 15
    max_frame_number = 45
    skip_frames = max_frame_number // n_frames_to_load
    print(f"Loading {n_frames_to_load} frames per video")
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
            for f in range(n_frames_to_load):
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
            # [15, 384, 384] -> [384, 384]
            prediction = torch.mean(prediction, dim=0)

            # downsample prediction by factor of 4
            prediction = torch.nn.functional.interpolate(
                prediction[None].unsqueeze(0),
                scale_factor=0.25,
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
