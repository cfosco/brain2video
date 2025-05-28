import os
import sys
import argparse
import json
import numpy as np

from eval_metrics import img_classify_metric, video_classify_metric
from eval_metrics import compute_metrics, ssim_metric_video

from utils import load_mp4_to_npy

sys.path.append('../')

def evaluate_embeddings(pred_folder, gt_folder, dataset='cc2017'):
    """
    Evaluate predicted vectors against ground truth vectors.
    """

    # Get test set
    test_embs = json.load(open(f'../data/metadata_{dataset}/{dataset}_test_set_video_paths.json'))
    test_embs = [f+'.npy' for f in test_embs]

    pred_folder_set = set(os.listdir(pred_folder))
    gt_folder_set = set(os.listdir(gt_folder))
    pred_embs = []
    gt_embs = []
    for f in test_embs:
        if f not in pred_folder_set:
            print(f"Warning: {f} not found in {pred_folder}. Skipping this file.")
        if f not in gt_folder_set:
            print(f"Warning: {f} not found in {gt_folder}. Skipping this file.")
        if f not in pred_folder_set or f not in gt_folder_set:
            continue

        pred_embs.append(np.load(os.path.join(pred_folder, f)))
        gt_embs.append(np.load(os.path.join(gt_folder, f)))

    # Compute metrics
    mse, mae, r2, corr_coef = compute_metrics(pred_embs, gt_embs)

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'corr_coef': corr_coef
    }


def evaluate_reconstructions(pred_folder, gt_folder, dataset='cc2017'):
    """
    Evaluate predicted reconstructions against ground truth reconstructions.
    """
    
    # Get test set
    test_vids = json.load(open(f'../data/metadata_{dataset}/{dataset}_test_set_video_paths.json'))
    test_vids = [f+'.mp4' for f in test_vids]

    pred_videos = [load_mp4_to_npy(os.path.join(pred_folder, f)) for f in test_vids] # if f in os.listdir(pred_folder)]
    
    # check number of frames in pred_videos and load same number of frames for gt_videos
    n_frames_in_pred = len(pred_videos[0])
    print(f"Number of frames in predicted videos: {n_frames_in_pred}")

    gt_videos = []
    gt_videos = [load_mp4_to_npy(os.path.join(gt_folder, f), frames_to_load=n_frames_in_pred, v=False) for f in test_vids]

    print(f"Loaded {len(pred_videos)} predicted videos and {len(gt_videos)} ground truth videos.")

    # Compute SSIM
    ssim_list = []
    for pred_video, gt_video in zip(pred_videos, gt_videos):
        ssim = ssim_metric_video(pred_video, gt_video)
        ssim_list.append(ssim)
    
    mean_ssim = np.mean(ssim_list)
    
    # Compute 2-way metrics
    acc_2way = img_classify_metric(
                pred_videos, 
                gt_videos,
                n_way=2,
                top_k=1,
                num_trials=100,
                cache_dir='.cache',
                device='cuda')
    
    # Compute 50-way metrics
    acc_50way = img_classify_metric(
                pred_videos, 
                gt_videos,
                n_way=50,
                top_k=1,
                num_trials=100,
                cache_dir='.cache',
                device='cuda')
    
    results = {
        'ssim': mean_ssim,
        '2_way_accuracy': np.mean(acc_2way),
        '50_way_accuracy': np.mean(acc_50way)
    }
    
    return results


def main(args):
    # Initialize empty dictionary for results
    results = {}

    if args.metrics_to_compute == 'all' or args.metrics_to_compute == 'vectors':
        # Compute metrics over predicted vectors (MSE, MAE, R2, Corr coef)
        vector_results = evaluate_embeddings(args.pred_folder, args.gt_folder, args.dataset)
        results.update(vector_results)

    if args.metrics_to_compute == 'all' or args.metrics_to_compute == 'reconstructions':
        # Compute metrics over reconstructions (SSIM, 5-way, 50-way)
        recon_results = evaluate_reconstructions(args.pred_folder, args.gt_folder, args.dataset)
        results.update(recon_results)

    # save combined eval results as json
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    output_path = os.path.join(args.save_folder, f'{os.path.basename(args.pred_folder)}_eval_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', 
                        '--pred_folder',
                        type=str, 
                        required=True,
                        help='Path to the folder containing predicted files to evaluate.')
    parser.add_argument('-gt', 
                        '--gt_folder',
                        type=str, 
                        required=True,
                        help='Path to the folder containing ground truth files to evaluate.')
    parser.add_argument('-metrics',
                        '--metrics_to_compute',
                        type=str, 
                        choices=['all', 'vectors', 'reconstructions'],
                        default='all',
                        help='Specify which metrics to compute: all, vectors, or reconstructions.')
    parser.add_argument('-d',
                        '--dataset',
                        type=str, 
                        required=True,
                        help='Dataset to evaluate on. One of bmd, cc2017')
    parser.add_argument('-save',
                        '--save_folder',
                        type=str, 
                        default='.',
                        help='Path to the folder where evaluation results will be saved.')


    args = parser.parse_args()

    main(args)