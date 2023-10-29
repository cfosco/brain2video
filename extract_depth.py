# Script to extract depth from the BOLDMoments videos

import os
import numpy as np
import cv2
import timm
import torch
import matplotlib.pyplot as plt
import argparse


def load_midas(model_type = "DPT_Hybrid"):

    # Load the midas depth estimator
    midas = torch.hub.load('intel-isl/MiDaS', model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    print(f"Loaded model: {model_type} in {device}")

    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    return midas, transform, device

def extract_depth_over_video_fbf(video_path, save_path):

    # Load the midas depth estimator
    midas, transform, device = load_midas()

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop over frames in the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # get frame number as int
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Convert the frame to RGB and apply transforms
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"Processing Frame {frame_num} / {frame_count}")
        # plt.imshow(frame)
        # plt.show()

        frame = transform(frame).to(device)

        # Run it through the model
        with torch.no_grad():
            prediction = midas(frame)

        # Convert the prediction to an image and save it
        prediction = prediction.squeeze().cpu().numpy()
        prediction = (255 * (prediction - prediction.min()) / (prediction.max() - prediction.min())).astype(np.uint8)

        # Save as png
        os.makedirs(os.path.join(save_path), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, f"{frame_num:03d}.png"), prediction)

        # plt.imshow(prediction)
        # plt.show()

    # Release the video file
    cap.release()


def extract_depth_over_frames(frames_path, save_path, midas, transform, device):

    # Loop over frames
    for frame_name in os.listdir(frames_path):

        # Load frame
        frame = cv2.imread(os.path.join(frames_path, frame_name))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        frame = transform(frame).to(device)

        # Run it through the model
        with torch.no_grad():
            prediction = midas(frame)

        # Convert the prediction to an image and save it
        prediction = prediction.squeeze().cpu().numpy()
        prediction = (255 * (prediction - prediction.min()) / (prediction.max() - prediction.min())).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, frame_name), prediction)



if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder_path", type=str, default=None, help="path to folder of videos")
    parser.add_argument("--frames_folder_path", type=str, default=None, help="path to folder of frame subfolders")
    parser.add_argument("--save_path", type=str, default=None, help="path to save depth as subfolders of images")

    args = parser.parse_args()
    video_folder_path = args.video_folder_path
    frames_folder_path = args.frames_folder_path
    save_path = args.save_path

    if video_folder_path is None and frames_folder_path is None:
        raise ValueError("Must provide either video_folder_path or frames_folder_path")
    
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "depth")

    if video_folder_path is not None:
        folder = video_folder_path
    else:
        folder = frames_folder_path


    midas, transform, device = load_midas()

    # Loop over videos in the folder
    for v in os.listdir(folder):
        print(f"Extracting depth from {v}...")
        f_p = os.path.join(folder, v)
        s_p = os.path.join(save_path, v)
        os.makedirs(s_p, exist_ok=True)
        extract_depth_over_frames(f_p, s_p, midas, transform, device)
    
    print("Done!")

    


    