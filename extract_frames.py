'''Functions to extract frames from videos'''

import os


def extract_frames_for_all_videos(video_path: str, output_path: str, fps: int = 15):
    """Extracts frames from all videos in a directory and saves them in a new directory.

    Args:
        video_path (str): Path to directory containing videos.
        output_path (str): Path to directory where frames will be saved.
        fps (int, optional): Frame rate. Defaults to 1.
    """
    for video in os.listdir(video_path):
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
    video_path = './data/10_videos_for_reconstruction_test'
    output_path = './data/10_videos_for_reconstruction_test/frames'
    extract_frames_for_all_videos(video_path, output_path)
