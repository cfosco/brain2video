"""Downloads the BOLDMoments data used for the brain reconstruction project"""

import os
import argparse


### Stimuli Download Functions


def download_videos_bmd(
    target_path="./data",
    csail_username="cfosco",
):
    path_to_videos = "/data/vision/oliva/datasets/BOLDMoments/prepared_data/metadata"

    print("Downloading BMD videos")

    # Download video data with scp
    os.system(
        f"scp {csail_username}@vision30.csail.mit.edu:{path_to_videos}/stimuli_mp4.tar.gz {target_path}"
    )

    # Extract the downloaded tar.gz file into target_path/stimuli_bmd
    os.system(f"tar -xzf {target_path}/stimuli_mp4.tar.gz -C {target_path}/stimuli_bmd")


def download_videos_had(
    target_path="./data",
    csail_username="cfosco",
):
    path_to_videos = "/data/vision/oliva/datasets/HumanActionsDataset/stimuli/*"
def download_videos_had(target_path='./data', csail_username='cfosco'):

    full_target_path = f"{target_path}/stimuli_had"
    os.makedirs(full_target_path, exist_ok=True)

    print("Downloading HAD videos")


    # Download video data with scp
    os.system(
        f"scp -r {csail_username}@vision30.csail.mit.edu:{path_to_videos} {full_target_path}"
    )


def download_images_nsd(
    target_path="./data",
    csail_username="cfosco",
):
    raise NotImplementedError

def download_images_nod(target_path='./data', csail_username='cfosco'):
    path_to_images = '/data/vision/oliva/scratch/datasets/NaturalObjectsDataset/stimuli/imagenet/*'

    full_target_path = f'{target_path}/stimuli_nod'
    os.makedirs(full_target_path, exist_ok=True)

    print("Downloading NOD images")

    # Download data with scp
    os.system(
        f'scp -r {csail_username}@vision30.csail.mit.edu:{path_to_images} {full_target_path}'
    )

### Betas Download Functions


## BMD
def download_betas_raw_bmd(
    target_path="./data", csail_username="cfosco", subjects_to_download=[1]
):
    for subject in subjects_to_download:
        path_to_data = f"/data/vision/oliva/blahner/BOLDMoments/preprocessed_data/responses/sub{subject:02d}/testing"

        full_target_path = f"{target_path}/betas_raw_bmd/sub{subject:02d}"
        os.makedirs(full_target_path, exist_ok=True)

        print("Downloading betas_raw for subject", subject)

        # Download fmri data with scp
        os.system(
            f"scp -r {csail_username}@vision30.csail.mit.edu:{path_to_data} {full_target_path}"
        )


def download_betas_glmsingle_impulse_bmd(
    target_path="./data", csail_username="cfosco", subjects_to_download=[1]
):
    for subject in subjects_to_download:
        path_to_data = f"/data/vision/oliva/datasets/BOLDMomentsGeneration/analysis/GLMMNI152_impulse/sub-{subject:02d}/GLMsingle/betas-prepared/prepared_allvoxel_pkl/"

        full_target_path = f"{target_path}/betas_impulse_bmd/sub{subject:02d}"
        os.makedirs(full_target_path, exist_ok=True)

        print("Downloading betas_impulse_bmd for subject", subject)

        # Download fmri data with scp
        os.system(
            f"scp -r {csail_username}@vision30.csail.mit.edu:{path_to_data} {full_target_path}"
        )


def download_betas_cifti_bmd(
    target_path="./data", csail_username="cfosco", subjects_to_download=[1]
):
    for subject in subjects_to_download:
        path_to_data = f'/data/vision/oliva/blahner/BMDGeneration/mindvis/data/BMD/cifti/task_videoGLM/sub-{subject:02d}/betas-prepared/prepared_allvoxel_pkl'

        full_target_path = f"{target_path}/betas_cifti_bmd/sub{subject:02d}"
        os.makedirs(full_target_path, exist_ok=True)

        print("Downloading betas_cifti_bmd for subject", subject)

        # Download fmri data with scp
        os.system(
            f"scp -r {csail_username}@vision30.csail.mit.edu:{path_to_data} {full_target_path}"
        )


## HAD
def download_betas_cifti_had(
    target_path="./data", csail_username="cfosco", subjects_to_download=[1]
):
    for subject in subjects_to_download:
        path_to_data = f'/data/vision/oliva/blahner/BMDGeneration/mindvis/data/HAD/cifti/task_videoGLM/sub-{subject:02d}/GLMsingle/betas-prepared/prepared_allvoxel_pkl'

        full_target_path = f"{target_path}/betas_cifti_had/sub{subject:02d}"
        os.makedirs(full_target_path, exist_ok=True)

        print("Downloading betas_cifti_had for subject", subject)

        # Download fmri data with scp
        os.system(
            f"scp -r {csail_username}@vision30.csail.mit.edu:{path_to_data} {full_target_path}"
        )


## NSD
def download_betas_glmsingle_impulse_nsd(
    target_path="./data", csail_username="cfosco", subjects_to_download=[1]
):
    # TODO finish this function
    raise NotImplementedError

## NOD
def download_betas_cifti_nod(target_path='./data', csail_username='cfosco', subjects_to_download=[1]):
    for subject in subjects_to_download:
        path_to_data = f'/data/vision/oliva/scratch/datasets/NaturalObjectsDataset/derivatives/HCP_rois/sub-{subject:02d}/*'

        full_target_path = f'{target_path}/betas_cifti_nod/sub{subject:02d}/prepared_allvoxel_pkl'
        os.makedirs(full_target_path, exist_ok=True)

        print("Downloading betas_cifti_nod for subject", subject)

        # Download fmri data with scp
        os.system(f'scp -r {csail_username}@vision30.csail.mit.edu:{path_to_data} {full_target_path}')


if __name__ == "__main__":
    AVAILABLE_DATA_TYPES = [
        'videos_bmd',
        'videos_had',
        'images_nod',
        'betas_raw_bmd',
        'betas_impulse_bmd',
        'betas_cifti_bmd',
        'betas_cifti_had',
        'betas_cifti_nod',
        # 'images_nsd',
        # 'betas_impulse_nsd',
    ]

    parser = argparse.ArgumentParser(description="Download BOLDMoments data")
    parser.add_argument(
        '-t',
        '--target_path',
        type=str,
        help="High level directory to download data to. Data Type and Subject folders will be created inside.",
        default="./data",
    )
    parser.add_argument(
        '-d',
        '--data_type',
        type=str,
        help=f'Type of data to download: one of {AVAILABLE_DATA_TYPES} or "all"',
        default="betas_cifti_bmd",
    )
    parser.add_argument(
        '-u',
        '--csail_username',
        type=str,
        help="CSAIL username to use for downloading",
        default="cfosco",
    )
    parser.add_argument(
        "-s",
        "--subjects_to_download",
        type=int,
        nargs="*",
        help="List of subjects to download fmri data for",
        default=[1],
    )

    args = parser.parse_args()

    if args.data_type not in AVAILABLE_DATA_TYPES + ["all"]:
        raise ValueError(f'data_type must be one of {AVAILABLE_DATA_TYPES} or "all"')

    print(args.subjects_to_download)
    if args.data_type == "videos_bmd" or args.data_type == "all":
        download_videos_bmd(args.target_path, args.csail_username)
    if args.data_type == "videos_had" or args.data_type == "all":
        download_videos_had(args.target_path, args.csail_username)
    if args.data_type == 'images_nod' or args.data_type == 'all':
        download_images_nod(args.target_path, args.csail_username)
    if args.data_type == 'betas_raw_bmd' or args.data_type == 'all':
        download_betas_raw_bmd(args.target_path, args.csail_username, args.subjects_to_download)
    if args.data_type == 'betas_impulse_bmd' or args.data_type == 'all':
        download_betas_glmsingle_impulse_bmd(args.target_path, args.csail_username, args.subjects_to_download)
    if args.data_type == 'betas_cifti_bmd' or args.data_type == 'all':
        download_betas_cifti_bmd(args.target_path, args.csail_username, args.subjects_to_download)
    if args.data_type == 'betas_cifti_had' or args.data_type == 'all':
        download_betas_cifti_had(args.target_path, args.csail_username, args.subjects_to_download)
    if args.data_type == 'betas_cifti_nod' or args.data_type == 'all':
        download_betas_cifti_nod(args.target_path, args.csail_username, args.subjects_to_download)
    # if args.data_type == 'betas_impulse_nsd' or args.data_type == 'all':
    #     download_betas_glmsingle_impulse_nsd(args.target_path, args.csail_username, args.subjects_to_download)
    # if args.data_type == 'images_nsd' or args.data_type == 'all':
    #     download_images_nsd(args.target_path, args.csail_username)
