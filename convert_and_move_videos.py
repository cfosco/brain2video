
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
        outputFile
        ]
    return commands_list


def convert_videos(mp4_dir, 
                   mp4_dir_converted, 
                   start_end_idx=[1000,1102]
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

    # convert videos
    os.makedirs(mp4_dir_converted, exist_ok=True)

    start_idx = start_end_idx[0]
    end_idx = start_end_idx[1]
    video_names = sorted(os.listdir(mp4_dir))
    for v in range(len(video_names)):
        if v < start_idx or v >= end_idx[1]:
            continue
        input = os.path.join(mp4_dir, video_names[v])
        output = os.path.join(mp4_dir_converted, video_names[v])    
        commands_list = build_command_list(user_input_dict, input, output)
        if subprocess.run(commands_list).returncode == 0:
            print ("FFmpeg Script Ran Successfully")
        else:
            raise Exception("There was an error running your FFmpeg script")
    
def move_to_nsf(path_to_videos, 
                save_path_in_nsf = ' /data/vision/oliva/scratch/ejosephs/brainGen_eval/stimuli',
                user = 'cfosco',
                remote = 'visiongpu52.csail.mit.edu',
                start_end_idx=[1000,1102]
                ):
    
    '''Scp the videos to the NSF server'''

    # make folder if it doesn't exist by running mkdir
    subprocess.run(['ssh', f'{user}@{remote}', 'mkdir', '-p', save_path_in_nsf])

    # run scp in a loop for all videos
    start_idx = start_end_idx[0]
    end_idx = start_end_idx[1]
    video_names = sorted(os.listdir(path_to_videos))
    for v in range(len(video_names)):
        if v < start_idx or v >= end_idx[1]:
            continue
        filename = video_names[v]
        if filename.endswith(".mp4") or filename.endswith(".gif"):
            print(filename)
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
                        default='/data/vision/oliva/scratch/ejosephs/brainGen_eval/stimuli',
                        help='path to save videos in NSF server')

    parser.add_argument('--start_end_idx',
                        required=False,
                        default=[1000,1102],
                        nargs="*",
                        type=int,
                        help="start and end idx of the videos for which to extract target vectors")


    args = parser.parse_args()

    convert_videos(args.mp4_dir, args.mp4_dir_converted, start_end_idx=args.start_end_idx)
    if args.to_nsf:
        move_to_nsf(args.mp4_dir_converted, args.save_path_in_nsf, start_end_idx=args.start_end_idx)