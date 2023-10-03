import glob
import os

import cv2
import matplotlib.pyplot as plt
import moten
import numpy as np
from PIL import Image
from scipy import misc

def evaluate_ME(gt, recon):
    assert(gt.shape == recon.shape)
    numFrames = gt.shape[0]

    correlation = []
    for f in range(numFrames):
        correlation.append(np.corrcoef(gt[f,:], recon[f,:])[0,1])
    
    return np.array(correlation)

if __name__ == '__main__':
    # Evaluate the motion energy (ME) (https://github.com/gallantlab/pymoten) of the reconstructed video against the ground truth video.
    # Evaluation is the correlation of the ME features of each frame averaged together. Features are temporally downsampled to whichever
    # video has the lowest frame rate, and correlations are averaged across time. Note that if this code is used for an encoding model of 
    # the BOLD signal, the features should be temporally downsampled to 1Hz as in Nishimoto et al., 2011.

    class arguments:
        def __init__(self) -> None:
            self.gt_me_root = "/mnt/t/BMDGeneration/analysis/metadata_features/motion_energy/BMD"
            self.recon_stim_root = "/mnt/t/BMDGeneration/analysis/metadata_features/motion_energy/BMDgeneral_sub01_blip_avgrepsFalse"
    args = arguments()

    save_root = os.path.join(args.recon_stim_root, "ME_feats")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    correlation_allvideo = []
    for v in range(102):
        videoID = f"vid_idx{v+1001:04d}"
        #load the ground truth ME features
        gt_feats = np.load(os.path.join(args.gt_me_root, f"{videoID}_MotionEnergy.npy"))
        gt_fps = int(gt_feats.shape[0]/3) #fps in seconds. assumes 3s video duration
        gt_numframes = gt_feats.shape[0]
        numFeats = gt_feats.shape[1]

        #load reconstructed ME features. Compute and save if they don't already exist
        if not os.path.exists(os.path.join(args.recon_stim_root, "ME_feats", f"{videoID}_MotionEnergy.npy")):
            #set up folders for saving
            plot_dir = os.path.join(save_root, "plots")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            #compute for mp4
            vid_path = os.path.join(args.recon_stim_root, f"{v+1001:04d}.mp4")
            vid_mp4 = cv2.VideoCapture(vid_path)
            assert(vid_mp4.isOpened())
            recon_fps = int(vid_mp4.get(cv2.CAP_PROP_FPS))
            numFrames = int(vid_mp4.get(cv2.CAP_PROP_FRAME_COUNT))
            assert(int(numFrames/3) == recon_fps) #need this assertion to assume a 3s stimuli duration later on
            luminance_images = moten.io.video2luminance(vid_path, size=(96,96), nimages=numFrames)
            nimages, vdim, hdim = luminance_images.shape
            assert(numFrames == nimages)
            pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=recon_fps)

            # Compute motion energy features
            recon_feats = pyramid.project_stimulus(luminance_images)
            recon_numframes = recon_feats.shape[0]

            np.save(os.path.join(args.recon_stim_root, "ME_feats", f"{videoID}_MotionEnergy.npy"), recon_feats)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.matshow(recon_feats, aspect='auto')
            plt.savefig(os.path.join(plot_dir, f"{videoID}_MEfeatures.png"))
            plt.clf()
            plt.close()

        else:
            recon_feats = np.load(os.path.join(args.recon_stim_root, "ME_feats", f"{videoID}_MotionEnergy.npy"))
            recon_numframes = recon_feats.shape[0]
            recon_fps = int(recon_numframes/3) #assumes a video length of 3s

        #temporally downsample to video with lower fps
        if recon_fps < gt_fps:
            gt_interp = np.zeros((recon_numframes, numFeats))
            for feat in range(numFeats):
                gt_interp[:,feat] = np.interp(np.linspace(0, gt_numframes, num=recon_numframes), np.arange(gt_numframes), gt_feats[:,feat]) #linear interpolation
            corr = evaluate_ME(gt_interp, recon_feats)
        elif gt_fps < recon_fps:
            recon_interp = np.zeros((recon_numframes, numFeats))
            for feat in range(numFeats):
                recon_interp[:, feat] = np.interp(np.linspace(0, recon_numframes, num=gt_numframes), np.arange(recon_numframes), recon_feats[:,feat]) #linear interpolation
            corr = evaluate_ME(gt_feats, recon_interp)
        else:
            corr = evaluate_ME(gt_feats, recon_feats)

        correlation_allvideo.append(np.mean(corr))
        print(f"average correlation across frames for video {videoID} is {np.mean(corr)}")
        np.save(os.path.join(save_root, f"{videoID}_MEcorrelation.npy"), corr)
    
    print("Done")