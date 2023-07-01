'''Downloads the BOLDMoments data used for the brain reconstruction project'''

import os
import argparse

def download_fmri(target_path, 
                  csail_username = 'cfosco',
                  path_to_fmri = '/data/vision/oliva/blahner/BOLDMoments/preprocessed_data/responses',
                  subjects_to_download = [1]):
    

    for subject in subjects_to_download:
        
        # Download fmri data with scp
        os.system(f'scp -r {csail_username}@visiongpu52.csail.mit.edu:{path_to_fmri}/sub0{subject}/testing {target_path}')
    

def download_videos(target_path,
                    csail_username = 'cfosco',
                    path_to_videos = '/data/vision/oliva/datasets/BOLDMoments/prepared_data/metadata',
                    ):
    
    # Download video data with scp

    print(f'scp {csail_username}@visiongpu52.csail.mit.edu:{path_to_videos}/stimuli_mp4.tar.gz {target_path}')

    os.system(f'scp {csail_username}@visiongpu52.csail.mit.edu:{path_to_videos}/stimuli_mp4.tar.gz {target_path}')


# Parse arguments for downloading either video or fmri
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download BOLDMoments data')
    parser.add_argument('--target_path', type=str, help='Path to download data to', default='./data')
    parser.add_argument('--data_type', type=str, help='Type of data to download: video, fmri or all', default='all')
    parser.add_argument('--csail_username', type=str, help='CSAIL username to use for downloading', default='cfosco')
    parser.add_argument('--path_to_fmri', type=str, help='Path to fmri data on CSAIL server', default='/data/vision/oliva/blahner/BOLDMoments/preprocessed_data/responses')
    parser.add_argument('--path_to_videos', type=str, help='Path to video data on CSAIL server', default='/data/vision/oliva/datasets/BOLDMoments/prepared_data/metadata')
    parser.add_argument('--subjects_to_download', type=list, help='List of subjects to download fmri data for', default=[1])

    args = parser.parse_args()

    if args.data_type == 'fmri':
        download_fmri(args.target_path, args.csail_username, args.path_to_fmri, args.subjects_to_download)
    elif args.data_type == 'video':
        download_videos(args.target_path, args.csail_username, args.path_to_videos)
    elif args.data_type == 'all':
        download_fmri(args.target_path, args.csail_username, args.path_to_fmri, args.subjects_to_download)
        download_videos(args.target_path, args.csail_username, args.path_to_videos)
    else:
        raise ValueError('Invalid data type. Must be either video or fmri')