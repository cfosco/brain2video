import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import pickle as pkl



def load_all_fmri_for_subject(path_to_subject_data: str) -> dict:
    """
    Load all fmri data for a subject into a dict with brain locations as keys
    """

    fmri_data = {}

    for p in os.listdir(path_to_subject_data):
        if '56789' not in p: continue
        
        brain_region = p.split('_')[0]

        with open(os.path.join(path_to_subject_data, p), 'rb') as f:
            data = pkl.load(f)
            fmri_data[brain_region] = data

    return fmri_data

def compute_tsne_embeddings(data: dict, perplexity: int = 50, n_iter: int = 3000, average_over: str = 'repetitions') -> np.ndarray:
    """
    Compute t-sne embeddings of the data
    """

    if average_over == 'videos':
        dim = 0
    elif average_over == 'repetitions':
        dim = 1
    elif average_over == 'voxels':
        dim = 2
    else:
        raise ValueError('average_over must be one of "videos", "repetitions" or "voxels"')

    avg_fmri = np.mean(data['train_data'], axis=dim)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(avg_fmri)

    return tsne_results

def t_sne_visualization(data: dict, perplexity: int = 50, n_iter: int = 3000, average_over: str = 'repetitions') -> None:
    """
    Plot t-sne visualization of the data
    """

    tsne_results = compute_tsne_embeddings(data, perplexity=perplexity, n_iter=n_iter, average_over=average_over)

    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('t-sne dim 1')
    plt.ylabel('t-sne dim 2')
    plt.show()



def transform_vids_to_gifs(path_to_vids='data/stimuli/mp4', path_to_gifs='data/stimuli/gif', size=128):
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.all import speedx
    from pygifsicle import optimize
    from PIL import Image

    os.makedirs(path_to_gifs, exist_ok=True)

    # Get list of all files
    all_files = os.listdir(path_to_vids)

    # Filter out the video files
    video_files = [file for file in all_files if file.endswith('.mp4')]

    # Process each video file
    for video_file in sorted(video_files):
        print(f'Processing {video_file}...')

        # Load the video
        video_clip = VideoFileClip(os.path.join(path_to_vids, video_file))

        # Speed up clip (reduces frames)
        video_clip = speedx(video_clip, factor=4)

        # Reduce resolution
        video_clip = video_clip.resize(height=size)
        
        # Get the video file name without extension
        video_name = os.path.splitext(video_file)[0]

        # Create a gif file path
        gif_file_path = os.path.join(path_to_gifs, f"{video_name}.gif")

        # Convert to gif and save
        video_clip.write_gif(gif_file_path, program='ffmpeg', opt='optimizeplus', fuzz=5)

        # Optimize the gif file
        # optimize(gif_file_path)


    print('All videos are converted to GIFs.')