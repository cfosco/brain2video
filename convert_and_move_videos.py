
import os
import subprocess
import argparse

def build_command_list(paramDict, inputFile, outputFile): 
    ffmpeg = "ffmpeg"
    commands_list = [
        ffmpeg,
        "-y",
        "-i",
        inputFile,
        "-c:v",
        paramDict["video_codec"],
        "-preset",
        paramDict["encoding_speed"],
        "-crf",
        paramDict["crf"],
        "-s",
        paramDict["frame_size"],
        "-c:a",
        paramDict["audio_codec"],
        "-b:a",
        paramDict["audio_bitrate"],
        "-ar",
        paramDict["sample_rate"],
        "-pix_fmt",
        "yuv420p",
        "-r",
        paramDict["frame_rate"],
        "-loglevel",
        paramDict["log_level"],
        outputFile
        ]
    return commands_list


def convert_videos(mp4_dir, 
                   mp4_dir_converted, 
                   start_end_idx=[1000,1102],
                   reconvert_if_exists=False
                   ):
    
    '''Convert videos to h264'''

    # set params
    user_input_dict = {}
    user_input_dict["video_codec"] = "libx264"
    user_input_dict["audio_codec"] =  "aac"
    user_input_dict["audio_bitrate"] = "196k"
    user_input_dict["sample_rate"] = "44100"
    user_input_dict["encoding_speed"] = "fast"
    user_input_dict["crf"] = "1"
    user_input_dict["frame_size"] = "360x360"
    user_input_dict["frame_rate"] = "8"
    user_input_dict["log_level"] = "error"

    # convert videos
    os.makedirs(mp4_dir_converted, exist_ok=True)

    start_idx = start_end_idx[0]
    end_idx = start_end_idx[1]
    video_names = sorted(os.listdir(mp4_dir))

    print("start_idx:", start_idx)
    print("end_idx:", end_idx)
    print("len(video_names):", len(video_names))
    for v in range(len(video_names)):
        video_idx = int(video_names[v].split('_')[0])
        if video_idx < start_idx or video_idx >= end_idx:
            continue

        input = os.path.join(mp4_dir, video_names[v])
        output = os.path.join(mp4_dir_converted, video_names[v])    

        # skip if video already exists
        if os.path.exists(output) and not reconvert_if_exists:
            continue

        commands_list = build_command_list(user_input_dict, input, output)
        print("Converting video:", video_names[v])
        if subprocess.run(commands_list).returncode == 0:
            print ("FFmpeg Script Ran Successfully")
        else:
            raise Exception("There was an error running your FFmpeg script")
    
def move_to_nsf(path_to_videos, 
                save_path_in_nsf = '/data/vision/oliva/scratch/ejosephs/brainGen_eval/stimuli/oracle_gens_zeroscope',
                user = 'cfosco',
                remote = 'oliva-titanrtx-1',
                start_end_idx=[1001,1103]
                ):
    
    '''Scp the videos to the NSF server'''

    # make folder if it doesn't exist by running mkdir
    subprocess.run(['ssh', f'{user}@{remote}', 'mkdir', '-p', save_path_in_nsf])

    # run scp in a loop for all videos
    start_idx = start_end_idx[0]
    end_idx = start_end_idx[1]
    video_names = sorted(os.listdir(path_to_videos))
    for v in range(len(video_names)):
        video_idx = int(video_names[v].split('_')[0])
        if video_idx < start_idx or video_idx >= end_idx:
            continue
        filename = video_names[v]
        if filename.endswith(".mp4") or filename.endswith(".gif"):
            print("Moving video to NSF:", filename)
            print("Command:", ['scp', os.path.join(path_to_videos, filename), 
                            f'{user}@{remote}:{save_path_in_nsf}'])
            subprocess.run(['scp', os.path.join(path_to_videos, filename), 
                            f'{user}@{remote}:{save_path_in_nsf}'])



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert videos to h264')
    parser.add_argument('--mp4_dir', 
                        type=str, 
                        default='./oracle_gens/mp4', 
                        help='path to mp4 directory')

    parser.add_argument('--mp4_dir_converted', 
                        type=str, 
                        default='./oracle_gens/mp4_h264', 
                        help='path to converted mp4 directory')

    parser.add_argument('--to_nsf', 
                        action='store_true',
                        help='move converted videos to NSF server')
    
    parser.add_argument('--save_path_in_nsf',
                        type=str,
                        default='/data/vision/oliva/scratch/ejosephs/brainGen_eval/stimuli/oracle_gens_zeroscope',
                        help='path to save videos in NSF server')

    parser.add_argument('--start_end_idx',
                        required=False,
                        default=[1001,1103],
                        nargs="*",
                        type=int,
                        help="start and end idx of the videos to convert and move")


    args = parser.parse_args()

    convert_videos(args.mp4_dir, args.mp4_dir_converted, start_end_idx=args.start_end_idx)
    if args.to_nsf:
        move_to_nsf(args.mp4_dir_converted, args.save_path_in_nsf, start_end_idx=args.start_end_idx)