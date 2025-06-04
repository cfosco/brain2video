import os
import pickle as pkl

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from himalaya.scoring import correlation_score
#from moviepy import VideoFileClip
from moviepy import * # Simple and nice, the __all__ is set in moviepy so only useful things will be loaded
from moviepy import VideoFileClip
#from moviepy.video.fx.all import speedx
#from moviepy import *
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

# from pygifsicle import optimize


def load_all_fmri_for_subject(path_to_subject_data: str) -> dict:
    """
    Load all fmri data for a subject into a dict with brain locations as keys
    """

    fmri_data = {}

    for p in os.listdir(path_to_subject_data):
        if "56789" not in p:
            continue

        brain_region = p.split("_")[0]

        with open(os.path.join(path_to_subject_data, p), "rb") as f:
            data = pkl.load(f)
            fmri_data[brain_region] = data

    return fmri_data


def compute_tsne_embeddings(
    data: dict,
    perplexity: int = 50,
    n_iter: int = 3000,
    average_over: str = "repetitions",
) -> np.ndarray:
    """
    Compute t-sne embeddings of the data
    """

    if average_over == "videos":
        dim = 0
    elif average_over == "repetitions":
        dim = 1
    elif average_over == "voxels":
        dim = 2
    else:
        raise ValueError(
            'average_over must be one of "videos", "repetitions" or "voxels"'
        )

    avg_fmri = np.mean(data["train_data"], axis=dim)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(avg_fmri)

    return tsne_results


def t_sne_visualization(
    data: dict,
    perplexity: int = 50,
    n_iter: int = 3000,
    average_over: str = "repetitions",
) -> None:
    """
    Plot t-sne visualization of the data
    """

    tsne_results = compute_tsne_embeddings(
        data, perplexity=perplexity, n_iter=n_iter, average_over=average_over
    )

    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel("t-sne dim 1")
    plt.ylabel("t-sne dim 2")
    plt.show()


def transform_vids_to_gifs(
    path_to_vids="data/stimuli/mp4",
    path_to_gifs="data/stimuli/gif",
    size=128,
    start_from=0,
):
    os.makedirs(path_to_gifs, exist_ok=True)

    # Get list of all files
    all_files = sorted(os.listdir(path_to_vids))

    # Filter out the video files
    video_files = [file for file in all_files if file.endswith(".mp4")]
    video_files = video_files[start_from:]

    # Process each video file
    for video_file in tqdm(sorted(video_files)):
        vid_to_gif(
            os.path.join(path_to_vids, video_file),
            os.path.join(path_to_gifs, video_file.replace(".mp4", ".gif")),
            size=size,
        )

    print("All videos are converted to GIFs.")


def frames_to_vid(frames, output_video_path, fps=30):
    frames = frames.squeeze()
    h, w, c = frames[0].shape

    # print("in frames_to_vid: frames type", frames.dtype)
    # print("in frames_to_vid: frames shape", frames.shape)
    # print("frames max and min", frames.max(), frames.min())
    
    def write_video(codec):
        video_writer = cv2.VideoWriter(
            output_video_path, codec, fps=fps, frameSize=(w, h)
        )

        # Check if video writer is initialized
        if not video_writer.isOpened():
            raise ValueError("Video writer could not be initialized.")

        for frame in frames:
            # Ensure the frame is in the correct format
            if frame.dtype != 'uint8':
                frame = (frame * 255).astype('uint8')
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # plt.imshow(img)
            # plt.show()
            # # save frame
            # plt.imsave('frame.png', img)
            # write to video_writer
            video_writer.write(img)
        video_writer.release()

    try:
        # First, attempt to use avc1 codec
        fourcc_avc1 = cv2.VideoWriter_fourcc(*"avc1")
        write_video(fourcc_avc1)
        print("Video written using avc1 codec.")
    except Exception as e:
        print(f"avc1 codec failed: {e}")
        try:
            # If avc1 fails, fall back to mp4v codec
            fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")
            write_video(fourcc_mp4v)
            print("Video written using mp4v codec.")
        except Exception as e:
            print(f"mp4v codec failed: {e}")
            try:
                # If mp4v also fails, fall back to MJPG codec
                fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
                write_video(fourcc_mjpg)
                print("Video written using MJPG codec.")
            except Exception as e:
                print(f"MJPG codec failed: {e}")
                raise ValueError("All codecs failed. Video could not be written.")


def vid_to_gif(video_filepath, output_gif_filepath, size=256):
    # print(f'Processing {video_filepath}...')

    # Load the video
    video_clip = VideoFileClip(video_filepath)
    video_clip.with_speed_scaled(factor=4)
    # Speed up clip (reduces frames)
    #video_clip = video_clip.fx(vfx.speedx, factor=4)
    #video_clip = speedx(video_clip, factor=4)
    # Reduce resolution
    video_clip = video_clip.resized(height=size)
    # Convert to gif and save
    video_clip.write_gif(
        output_gif_filepath,
        program="ffmpeg",
        opt="optimizeplus",
        fuzz=5,
        # verbose=False,
        logger=None,
    )

    # Optimize the gif file
    # optimize(gif_file_path)


def prompt_list_from_boldmoments_annots(annots):
    # for i, a in annots.items():
    #         for c in range(len(a['text_descriptions'])):
    #             prompt = a['text_descriptions'][c]
    pass


def load_frames_to_npy(frame_folder, step_frames=1, size=256):
    """Loads frames with cv2.imread and returns them as a numpy array.
    Frames are in BGR format.

    Args:
    - frame_folder (str): Path to the folder containing the frames.
    - step_frames (int): step size for loading frames. 1 means that every frame is loaded. 2 means that we skip one frame in between each loaded frame.
    """
    frames = []
    frame_filenames = os.listdir(frame_folder)
    for i in range(0, len(frame_filenames), step_frames):
        frame_path = os.path.join(frame_folder, frame_filenames[i])
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)

    return np.stack(frames)

def load_mp4_to_npy(video_path, step_frames=1, frames_to_load=None, size=256, v=False):
    """
    Loads frames from a video file by reading all available frames and then sampling.
    
    Args:
        video_path (str): Path to the video file.
        step_frames (int): Legacy parameter, kept for backward compatibility.
        frames_to_load (int): Number of frames to return after sampling.
        size (int): Size to resize frames to (size x size).
        v (bool): Verbose mode for debugging.
    
    Returns:
        np.ndarray: Array of loaded frames with shape (num_frames, height, width, channels)
    """
    import os
    import cv2
    import numpy as np
    
    if v:
        print(f"Loading video: {video_path}")
    
    # Check file existence
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}. Check if OpenCV supports the codec.")
    
    # Read all frames first
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB and resize
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            all_frames.append(frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    cap.release()
    
    if len(all_frames) == 0:
        raise ValueError(f"Failed to read any frames from {video_path}")
    
    if v:
        print(f"Successfully read {len(all_frames)} frames from video")
    
    # Sample frames if needed
    if frames_to_load is not None:
        if frames_to_load > len(all_frames):
            if v:
                print(f"Warning: Requested {frames_to_load} frames but video only has {len(all_frames)}")
            
            # Duplicate last frame if we don't have enough
            while len(all_frames) < frames_to_load:
                all_frames.append(all_frames[-1])
                
            if v:
                print(f"Duplicated last frame to reach {len(all_frames)} frames")
        
        elif frames_to_load < len(all_frames):
            # Uniform sampling to get exactly frames_to_load frames
            indices = np.linspace(0, len(all_frames)-1, frames_to_load, dtype=int)
            wanted_frames = [all_frames[i] for i in indices]
            
            if v:
                print(f"Sampled {frames_to_load} frames uniformly from {len(all_frames)} frames")
    
    else:
        # use step_frames to sample frames
        wanted_frames = all_frames[::step_frames]
        if v:
            print(f"Sampled {len(wanted_frames)} frames with step size {step_frames} from {len(all_frames)} frames")

    return np.stack(wanted_frames)


def load_frames_to_tensor(root_dir, batch_size=8, n_frames_to_load=45, size=268):
    """
    Load image frames from subfolders into a torch tensor.

    Args:
    - root_dir (str): Path to the root directory containing subfolders of frames.

    Returns:
    - Tensor of shape (b, f, c, h, w) where:
      b: number of videos (subfolders)
      f: number of frames per video (45 for bold moments)
      c: number of channels (3 for RGB images)
      h: height of the video
      w: width of the video
    """

    # List all subfolders (videos)
    video_folders = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]

    # Sort the folders for consistency
    video_folders.sort()

    # Placeholder list to store all video tensors
    all_videos = []

    for video_folder in video_folders:
        video_path = os.path.join(root_dir, video_folder)

        # List all frames in the current video folder
        frames = [
            f for f in os.listdir(video_path) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # Sort the frames for consistency
        frames.sort()

        # Placeholder list to store all frame tensors for the current video
        video_frames = []

        for frame in frames:
            frame_path = os.path.join(video_path, frame)

            # Load the image using PIL and convert to RGB
            img = Image.open(frame_path).convert("RGB")

            # Convert the PIL image to a torch tensor
            tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

            video_frames.append(tensor)

        # Stack all frame tensors for the current video
        video_tensor = torch.stack(video_frames, dim=0)
        all_videos.append(video_tensor)

    # Stack all video tensors
    result_tensor = torch.stack(all_videos, dim=0)

    return result_tensor


def preprocess_video_for_vid2vid_pipeline(video):
    supported_formats = (np.ndarray, torch.Tensor, PIL.Image.Image)

    if isinstance(video, supported_formats):
        video = [video]
    elif not (
        isinstance(video, list) and all(isinstance(i, supported_formats) for i in video)
    ):
        raise ValueError(
            f"Input is in incorrect format: {[type(i) for i in video]}. Currently, we only support {', '.join(supported_formats)}"
        )

    if isinstance(video[0], PIL.Image.Image):
        video = [np.array(frame) for frame in video]

    if isinstance(video[0], np.ndarray):
        video = (
            np.concatenate(video, axis=0)
            if video[0].ndim == 5
            else np.stack(video, axis=0)
        )

        if video.dtype == np.uint8:
            video = np.array(video).astype(np.float32) / 255.0

        if video.ndim == 4:
            video = video[None, ...]

        video = torch.from_numpy(video.transpose(0, 4, 1, 2, 3))

    elif isinstance(video[0], torch.Tensor):
        video = (
            torch.cat(video, axis=0)
            if video[0].ndim == 5
            else torch.stack(video, axis=0)
        )

        # don't need any preprocess if the video is latents
        channel = video.shape[1]
        if channel == 4:
            return video

        # move channels before num_frames
        video = video.permute(0, 2, 1, 3, 4)

    # normalize video
    video = 2.0 * video - 1.0

    return video


def save_vectors_npy(vectors, save_path, filenames):
    os.makedirs(save_path, exist_ok=True)

    if type(vectors) == torch.Tensor:
        ran = range(vectors.shape[0])
    else:
        ran = range(len(vectors))

    # Save each target vector for each video as its own npy file
    for i in ran:
        fi = os.path.splitext(filenames[i])[0]
        name = os.path.join(save_path, f"{fi}.npy")
        os.makedirs(os.path.dirname(name), exist_ok=True)
        np.save(os.path.join(save_path, f"{fi}.npy"), vectors[i])


def print_current_gpu_memory():
    t = torch.cuda.get_device_properties(0).total_memory / 1024**2
    r = torch.cuda.memory_reserved(0) / 1024**2
    a = torch.cuda.memory_allocated(0) / 1024**2
    f = r - a  # free inside reserved

    print(
        f"GPU Mem -- Total: {int(t)} MB, Reserved: {int(r)}, Allocated: {int(a)}, Free: {int(f)}"
    )


def norm_and_transp(img, minus_1_to_1=True):
    if minus_1_to_1:
        return (np.array(img).transpose(2, 0, 1) / 255.0) * 2.0 - 1.0
    else:
        return np.array(img).transpose(2, 0, 1) / 255.0


### PLOTTING FUNCTIONS


def plot_video(video, frames_to_skip=1):
    """
    Plot a video with a subplot for each frame, skipping frames_to_skip frames
    """
    plt.figure(figsize=(10, 10))
    for i, frame in enumerate(video[::frames_to_skip]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(frame)
        plt.axis("off")
    plt.show()
