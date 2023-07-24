'''Functions to extract frames from videos'''

import os
import argparse
from tqdm import tqdm


def extract_frames_for_all_videos(video_path: str, output_path: str, fps: int = 15):
    """Extracts frames from all videos in a directory and saves them in a new directory.

    Args:
        video_path (str): Path to directory containing videos.
        output_path (str): Path to directory where frames will be saved.
        fps (int, optional): Frame rate. Defaults to 1.
    """

    print(f"Extracting frames from videos in {video_path} and saving them in {output_path}.")

    for video in tqdm(os.listdir(video_path)):
        if video.endswith('.mp4'):
            video_name = video.split('.')[0]
            vid_p = os.path.join(video_path, video)
            o_p = os.path.join(output_path, video_name)
            extract_frames(vid_p, o_p, fps)


def extract_frames(video_path: str, output_path: str, fps: int = 1):
    """Extracts frames from a video and saves them in a new directory.

    Args:
        video_path (str): Path to video.
        output_path (str): Path to directory where frames will be saved.
        fps (int, optional): Frame rate. Defaults to 1.
    """
    os.makedirs(output_path, exist_ok=True)
    os.system(f'ffmpeg -i {video_path} -vf fps={fps} {output_path}/%03d.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_path",
        type=str,
        help="path to folder of videos to extract frames from",
        default='./data/stimuli/mp4',
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="path to folder where frames will be saved",
        default='./data/stimuli/frames',
    )

    args = parser.parse_args()
    extract_frames_for_all_videos(args.video_path, args.output_path)
