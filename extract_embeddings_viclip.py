# TODO: DEPRECATE AND MOVE LOGIC TO extract_embeddings.py

import sys
sys.path.append('../')
sys.path.append('../ViCLIP')
from viclip import ViCLIP
from simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
import cv2
import torch
import os
import argparse
from utils import load_frames_to_npy


def main(args):
    device = args.device
    tokenizer = _Tokenizer()
    vclip = ViCLIP(tokenizer)
    vclip = vclip.to(device)

    plot = False # For debugging purposes

    for frame_folder in sorted(os.listdir(args.path_to_frames)):
        frames = load_frames_to_npy(os.path.join(args.path_to_frames, frame_folder))
        if plot:
            plot_frames(frames)
        frames_tensor = frames_to_tensor_for_viclip(frames, device=device)
        emb = vclip.get_vid_features(frames_tensor)
        save_embedding(os.path.join(args.save_path, frame_folder+'.npy'), emb)


def save_embedding(save_path, emb):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, emb.cpu().numpy().squeeze())

def normalize(data):
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    return (data/255.0-v_mean)/v_std

def frames_to_tensor_for_viclip(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]

    # Resize and switch from BGR to RGB
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]

    # Add batch dimension and normalize with ImageNet stats
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)

    # Transpose to (batch, frames, channels, height, width)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    # plot_vid_tube(vid_tube)

    # Convert to torch tensor
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def plot_frames(frames):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(frames))
    for i, frame in enumerate(frames):
        axs[i].imshow(frame)
    plt.show()

def plot_vid_tube(vid_tube):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, vid_tube.shape[1])
    for i, frame in enumerate(vid_tube[0]):
        axs[i].imshow(np.transpose(frame,(1,2,0)))
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_frames",
        required=False,
        type=str,
        default='./data/stimuli/frames',
        help="Path to the videos for which to extract target vectors",
    )

    parser.add_argument(
        "--save_path",
        required=False,
        type=str,
        default='./data/viclip_embeddings_bmd',
        help="Path to save the extracted features",
    )

    parser.add_argument(
        "--device",
        required=False,
        type=str,
        default='cuda',
        help="Device to run ViCLIP on",
    )


    args = parser.parse_args()
    main(args)