"""Script to reconstruct videos from predicted 
latent and conditioning vectors using Zeroscope"""


import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from diffusers import VideoToVideoSDPipeline, DPMSolverMultistepScheduler
from einops import rearrange
from utils import vid_to_gif, frames_to_vid
import sys
from dataset import BMDReconstructionDataset

sys.path.append("./blip/models")
from blip import blip_decoder


def main(args):
    # Follow vid2vid approach: https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py

    # Load text2vid pipeline
    pipe = VideoToVideoSDPipeline.from_pretrained(
        "../zeroscope_v2_576w", torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # Load BLIP model
    print("Loading BLIP model")
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    blip_deco = blip_decoder(pretrained=model_url, image_size=240, vit="base")
    blip_deco.eval()
    blip_deco = blip_deco.to("cuda")

    if args.use_gt_vecs or args.use_gt_z:
        args.z_path = "./data/target_vectors/z_zeroscope"
    if args.use_gt_vecs or args.use_gt_blip:
        args.blip_path = "./data/target_vectors/blip"

    z_blip_dataset = BMDReconstructionDataset(
        {"z": args.z_path, "blip": args.blip_path},
        args.set,
        rearrange_funcs={
            "z": lambda z: rearrange(z, '(f c w h) -> c f w h', f=15, c=4, w=33, h=33),
            "blip": lambda bl: rearrange(bl, '(k l) -> k l', k=226, l=768)
        },
    )

    # z_blip_dataloader = DataLoader(z_blip_dataset, batch_size=1, shuffle=False, num_workers=0)

    # if args.set == 'train':
    #     filename = 'preds_train.npy'
    #     idx_start = 1
    # elif args.set == 'test':
    #     filename = 'preds_test.npy'
    #     idx_start = 1001

    # Load predicted latents
    # if args.use_gt_vecs or args.use_gt_z:
    #     print("Using ground truth latents")
    #     z_path = './data/target_vectors/z_zeroscope'
    #     z = [np.load(os.path.join(z_path, f'{n:04d}.npy')) for n in range(idx_start,len(os.listdir(z_path))+1)]
    #     z = np.array(z)
    # else:
    #     z = np.load(os.path.join(args.z_path, filename))

    # Reshape latents to expected shape: (b c f w h) with c = 4, f = 15
    # z = rearrange(z, 'b (f c w h) -> b c f w h', f=15, c=4, w=33, h=33)
    # z = torch.tensor(z).float().cuda()
    # print('z.shape',z.shape)

    # # Load conditioning vectors
    # if args.use_gt_vecs or args.use_gt_c:
    #     c_path = './data/target_vectors/c_zeroscope'
    #     c = [np.load(os.path.join(c_path, f'{n:04d}.npy')) for n in range(idx_start,len(os.listdir(c_path))+1)]
    #     c = np.array(c)
    # else:
    #     c = np.load(os.path.join(args.c_path, filename))

    # # Reshape cond vectors
    # c = rearrange(c, 'b (k l) -> b k l', k=77)
    # c = torch.tensor(c).float().cuda()
    # print('c.shape', c.shape)

    # Load BLIP vectors
    # if args.use_gt_vecs or args.use_gt_blip:
    #     blip_path = './data/target_vectors/blip'
    #     blip_embeds = [np.load(os.path.join(blip_path, f'{n:04d}.npy')) for n in range(idx_start,len(os.listdir(blip_path))+1)]
    #     blip_embeds = np.array(blip_embeds)
    # else:
    #     blip_embeds = np.load(os.path.join(args.blip_path, filename))

    # # Reshape BLIP vectors
    # blip_embeds = rearrange(blip_embeds, 'b (k l) -> b k l', k=226, l=768)
    # blip_embeds = torch.tensor(blip_embeds).float().cuda()
    # print('blip_embeds.shape', blip_embeds.shape)

    # captions = []

    # for i in range(len(z)):
    for z, blip_emb, filename in z_blip_dataset:
        z = z.cuda()
        blip_emb = blip_emb.cuda()
        filename = filename.split(".")[0]

        # Get captions
        print("Generating captions")
        caption = embeds_to_captions(
            blip_deco,
            device="cuda",
            image_embeds=blip_emb[None],
            num_beams=4,
            max_length=20,
            min_length=4,
            repetition_penalty=6.0,
        )

        print("Generated caption for video", filename, ":", caption)

        # captions.append(caption)
        # Save caption as txt
        txt_path = os.path.join(args.output_path, "txt")
        os.makedirs(txt_path, exist_ok=True)
        with open(os.path.join(txt_path, f"{filename}_{caption[0]}.txt"), "w") as f:
            f.write(caption[0])

        # Reconstruct videos
        print("Reconstructing video", filename)
        rec_vid_frames = pipe(
            prompt=caption,
            video=z[None] * args.latent_factor,
            num_inference_steps=50,
            strength=0.5,  # Strength controls the noise applied to the latent before starting the diffusion process. Higher strength = higher noise. Acts as a % of inference steps
        ).frames

        print(
            "rec_vid_frames len",
            len(rec_vid_frames),
            "rec_vid_frames[0] shape",
            rec_vid_frames[0].shape,
        )

        # Save reconstructed videos as mp4 and gif
        vid_path = os.path.join(args.output_path, "mp4")
        os.makedirs(vid_path, exist_ok=True)
        gif_path = os.path.join(args.output_path, "gif")
        os.makedirs(gif_path, exist_ok=True)

        frames_to_vid(rec_vid_frames, os.path.join(vid_path, f"{filename}.mp4"), fps=5)
        vid_to_gif(
            os.path.join(vid_path, f"{filename}.mp4"),
            os.path.join(gif_path, f"{filename}.gif"),
            size=264,
        )


def embeds_to_captions(
    model,
    device,
    image_embeds,
    sample=False,
    num_beams=3,
    max_length=30,
    min_length=10,
    top_p=0.9,
    repetition_penalty=1.0,
):
    if not sample:
        image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    model_kwargs = {
        "encoder_hidden_states": image_embeds,
        "encoder_attention_mask": image_atts,
    }

    prompt = [model.prompt]
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_ids[:, 0] = model.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1]

    # print("input_ids.shape", input_ids.shape)
    # print("prompt", prompt)

    if sample:
        # nucleus sampling
        outputs = model.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=model.tokenizer.sep_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            **model_kwargs,
        )
    else:
        # beam search
        outputs = model.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            eos_token_id=model.tokenizer.sep_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            **model_kwargs,
        )

    captions = []
    for output in outputs:
        caption = model.tokenizer.decode(output, skip_special_tokens=True)
        captions.append(caption[len(model.prompt) :])
    return captions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reconstruct videos from predicted latents and conditioning vectors"
    )

    parser.add_argument(
        "--z_path",
        type=str,
        help="Path to predicted latents npy file",
        default="./estimated_vectors/z_zeroscope/regressor:swigluwithscheduleronval_hidden:2048_fmritype:betas_impulse_rois:BMDgeneral_avgtrainreps:False_usensd:False_sub[1]_z_zeroscope_unflattened",
    )

    parser.add_argument(
        "--blip_path",
        type=str,
        help="Path to predicted conditioning vectors npy file",
        default=None
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for reconstructed videos",
        default="./reconstructions/BMDgeneral_sub01_test3",
    )

    parser.add_argument(
        "--set",
        type=str,
        help="Set to reconstruct videos from. Options: train, test",
        default="test",
    )

    parser.add_argument(
        "--use_gt_vecs",
        action="store_true",
        help="Whether to use ground truth conditioning vectors",
    )

    parser.add_argument(
        "--use_gt_z",
        action="store_true",
        help="Whether to use ground truth z vectors",
    )

    parser.add_argument(
        "--use_gt_blip",
        action="store_true",
        help="Whether to use ground truth blip vectors",
    )

    parser.add_argument(
        "--latent_factor",
        type=float,
        help="Factor to scale the predicted latents by",
        default=3.0,
    )

    args = parser.parse_args()

    main(args)
