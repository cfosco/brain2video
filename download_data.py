'''Downloads the BOLDMoments data used for the brain reconstruction project'''

import os
import argparse

def download_betas_raw(target_path = './data', 
                  csail_username = 'cfosco',
                  subjects_to_download = [1]
                  ):
    

    for subject in subjects_to_download:
        path_to_data = f'/data/vision/oliva/blahner/BOLDMoments/preprocessed_data/responses/sub{subject:02d}/testing',

        os.makedirs(f'{target_path}/betas_raw/sub0{subject}', exist_ok=True)
        
        print("Downloading betas_raw for subject", subject)

        # Download fmri data with scp
        os.system(f'scp -r {csail_username}@visiongpu52.csail.mit.edu:{path_to_data} {target_path}/betas_raw/sub0{subject}')
    

def download_betas_glmsingle_impulse(target_path='./data',
                                    csail_username = 'cfosco',
                                    subjects_to_download = [1]
                                    ):

    
    for subject in subjects_to_download:
        path_to_data = f'/data/vision/oliva/datasets/BOLDMomentsGeneration/analysis/GLMMNI152_impulse/sub-{subject:02d}/GLMsingle/betas-prepared/prepared_allvoxel_pkl/'

        os.makedirs(f'{target_path}/betas_impulse/sub0{subject}', exist_ok=True)

        print("Downloading betas_impulse for subject", subject)
            
        # Download fmri data with scp
        os.system(f'scp -r {csail_username}@visiongpu52.csail.mit.edu:/{path_to_data} {target_path}/betas_impulse/sub0{subject}')


def download_videos(target_path = './data',
                    csail_username = 'cfosco',
                    ):
    
    path_to_videos = '/data/vision/oliva/datasets/BOLDMoments/prepared_data/metadata',

    print("Downloading videos")

    # Download video data with scp
    os.system(f'scp {csail_username}@visiongpu52.csail.mit.edu:{path_to_videos}/stimuli_mp4.tar.gz {target_path}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download BOLDMoments data')
    parser.add_argument('--target_path', type=str, help='High level directory to download data to. Data Type and Subject folders will be created inside.', default='./data')
    parser.add_argument('--data_type', type=str, help='Type of data to download: video, betas_raw, betas_impulse or all', default='all')
    parser.add_argument('--csail_username', type=str, help='CSAIL username to use for downloading', default='cfosco')
    parser.add_argument('--subjects_to_download', type=int, nargs="*", help='List of subjects to download fmri data for', default=[1])

    args = parser.parse_args()

    if args.data_type not in ['video', 'betas_raw', 'betas_impulse', 'all']:
        raise ValueError('data_type must be one of video, betas_raw, betas_impulse or all')

    print(args.subjects_to_download)

    if args.data_type == 'betas_raw' or args.data_type == 'all':
        download_betas_raw(args.target_path, args.csail_username, args.subjects_to_download)
    if args.data_type == 'betas_impulse' or args.data_type == 'all':
        download_betas_glmsingle_impulse(args.target_path, args.csail_username, args.subjects_to_download)
    if args.data_type == 'video' or args.data_type == 'all':
        download_videos(args.target_path, args.csail_username)