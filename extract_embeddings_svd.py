import argparse
import os

import numpy as np
import torch
from diffusers import DiffusionPipeline

from diffusers import StableVideoDiffusionPipeline
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    pipeline,
)

from dataset import DATASET_OUTPUT_TYPES, DATASET_PATHS, STIM_DATASET_MAP
from svd import CustomStableVideoDiffusionPipeline


def main(args):
    dataset = STIM_DATASET_MAP[args.dataset](
        args.input_path
        if args.input_path is not None
        else DATASET_PATHS[args.dataset]["stimuli"],
        args.metadata_path
        if args.metadata_path is not None
        else DATASET_PATHS[args.dataset]["metadata"],
        subset="all",
        resolution=244,
        transform=None,
        normalize=False,
        return_filename=True,
        load_from_frames=args.load_from_frames,
        skip_frames=None,
        n_frames_per_video=8,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=1
    )

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Device:", device)

    if (
        args.emb_wanted == "zeroscope_c"
        and DATASET_OUTPUT_TYPES[args.dataset] == "text"
    ):
        get_emb = ZeroscopeTextEmbeddingPipeline(device)
        out_folder = "zeroscope_emb_77x1024"  # TODO: Double check that this is correct
    elif (
        args.emb_wanted == "zeroscope_c"
        and DATASET_OUTPUT_TYPES[args.dataset] == "image"
    ):
        get_emb = ZeroscopeImageEmbeddingPipeline(device, args.caption_first)
        out_folder = "zeroscope_emb_77x1024"
    elif (
        args.emb_wanted == "zeroscope_c"
        and DATASET_OUTPUT_TYPES[args.dataset] == "video"
    ):
        get_emb = ZeroscopeVideoEmbeddingPipeline(device, args.caption_first)
        out_folder = "zeroscope_emb_77x1024"
    elif args.emb_wanted == "clip_c" and DATASET_OUTPUT_TYPES[args.dataset] == "image":
        get_emb = CLIPImageEmbeddingPipeline(device)
        out_folder = "clip_emb_257x1024"
    elif args.emb_wanted == "clip_c" and DATASET_OUTPUT_TYPES[args.dataset] == "video":
        get_emb = CLIPVideoEmbeddingPipeline(device)
        out_folder = "clip_emb_257x1024"

    elif args.emb_wanted == "svd_emb" and DATASET_OUTPUT_TYPES[args.dataset] == "video":
        get_emb = SVDVideoEmbeddingPipeline(device)
        out_folder = "svd_emb_1024"

    elif (
        args.emb_wanted == "svd_emb_lats"
        and DATASET_OUTPUT_TYPES[args.dataset] == "video"
    ):
        get_emb = SVDVideoEmbeddingAndLatentsPipeline(device)
        out_folder = "svd_emb_lats_1024"

    if args.save_path is None:
        args.save_path = os.path.join(
            f"./data/target_vectors_{args.dataset}", out_folder
        )

    for stims, filenames in tqdm(dataloader):
        print("Processing filenames:", filenames)

        # print("stims.shape", stims.shape)
        # print("stims.type", type(stims))
        # print("stims max min", stims.max(), stims.min())

        embeddings = get_emb(stims)
        for embedding, filename in zip(embeddings, filenames):
            filename = os.path.splitext(filename)[0]
            print(
                "Saving embedding for",
                filename,
                "at",
                os.path.join(args.save_path, f"{filename}.npy"),
            )
            save_embedding(os.path.join(args.save_path, f"{filename}.npy"), embedding)


def main_alt(args):
    dataset = STIM_DATASET_MAP[args.dataset](
        args.input_path
        if args.input_path is not None
        else DATASET_PATHS[args.dataset]["stimuli"],
        args.metadata_path
        if args.metadata_path is not None
        else DATASET_PATHS[args.dataset]["metadata"],
        subset="all",
        resolution=244,
        transform=None,
        normalize=False,
        return_filename=True,
        load_from_frames=args.load_from_frames,
        skip_frames=None,
        n_frames_per_video=1,
    )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=16, shuffle=False, num_workers=1
    # )

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Device:", device)

    if (
        args.emb_wanted == "zeroscope_c"
        and DATASET_OUTPUT_TYPES[args.dataset] == "text"
    ):
        get_emb = ZeroscopeTextEmbeddingPipeline(device)
        out_folder = "zeroscope_emb_77x1024"  # TODO: Double check that this is correct
    elif (
        args.emb_wanted == "zeroscope_c"
        and DATASET_OUTPUT_TYPES[args.dataset] == "image"
    ):
        get_emb = ZeroscopeImageEmbeddingPipeline(device, args.caption_first)
        out_folder = "zeroscope_emb_77x1024"
    elif (
        args.emb_wanted == "zeroscope_c"
        and DATASET_OUTPUT_TYPES[args.dataset] == "video"
    ):
        get_emb = ZeroscopeVideoEmbeddingPipeline(device, args.caption_first)
        out_folder = "zeroscope_emb_77x1024"
    elif args.emb_wanted == "clip_c" and DATASET_OUTPUT_TYPES[args.dataset] == "image":
        get_emb = CLIPImageEmbeddingPipeline(device)
        out_folder = "clip_emb_257x1024"
    elif args.emb_wanted == "clip_c" and DATASET_OUTPUT_TYPES[args.dataset] == "video":
        get_emb = CLIPVideoEmbeddingPipeline(device)
        out_folder = "clip_emb_257x1024"

    elif args.emb_wanted == "svd_emb" and DATASET_OUTPUT_TYPES[args.dataset] == "video":
        get_emb = SVDVideoEmbeddingPipeline(device)
        out_folder = "svd_emb_1024"

    elif (
        args.emb_wanted == "svd_emb_lats"
        and DATASET_OUTPUT_TYPES[args.dataset] == "video"
    ):
        get_emb = SVDVideoEmbeddingAndLatentsPipeline(device)
        out_folder = "svd_emb_lats_1024"

    elif (
        args.emb_wanted == "svd_emb_lats_all_frames"
        and DATASET_OUTPUT_TYPES[args.dataset] == "video"
    ):
        get_emb = SVDVideoEmbeddingAndLatentsPipeline(device)
        out_folder = "svd_emb_lats_all_frames"

    if args.save_path is None:
        args.save_path = os.path.join(
            f"./data/target_vectors_{args.dataset}", out_folder
        )

    import random

    import imageio

    random.shuffle(dataset.vid_paths)
    dataloader = enumerate(reversed(dataset))
    for idx, (stim, filename) in tqdm(dataloader):
        # print("Processing filenames:", filenames)
        print("Processing filename:", filename)

        print("stims.shape", stim.shape)
        print("stims.type", type(stim))
        print("stims max min", stim.max(), stim.min())
        video_path = os.path.join(dataset.path, dataset.vid_paths[idx])
        # print("Loading video", video_path)
        if not video_path.endswith(".mp4"):
            video_path = f"{video_path}.mp4"

        # load middle frame as pil image:
        print(f"video_path: {video_path}")
        embeddings_path = os.path.join(args.save_path, f"{filename}.npy")
        if os.path.exists(embeddings_path):
            print("Embedding already exists for", filename)
            continue

        all_embs = []
        reader = imageio.get_reader(video_path, "ffmpeg")
        for i, frame in enumerate(reader):
            if i % 4 != 0:
                continue
            stim = Image.fromarray(frame)
            embedding = get_emb(stim)
            all_embs.append(embedding)

        embedding = torch.cat(all_embs)
        filename = os.path.splitext(filename)[0]
        print("Saving embedding for", filename, "at", embeddings_path)
        save_embedding(embeddings_path, embedding)


def load_middle_frame(video_path: str) -> Image.Image:
    import imageio

    vid = imageio.get_reader(video_path, "ffmpeg")
    return Image.fromarray(vid.get_data(0))


class CLIPImageEmbeddingPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)

    def __call__(self, img_batch):
        img_batch = img_batch.to(self.device)

        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(images=img_batch, return_tensors="pt")
            inputs.pixel_values = inputs.pixel_values.to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        return embeddings


class CLIPVideoEmbeddingPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)

    def __call__(self, video_batch):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for v in video_batch:
                inputs = self.processor(images=v, return_tensors="pt")
                inputs["pixel_values"] = inputs.pixel_values.to(self.device)
                outputs = self.model(**inputs)

                # Average over frames
                emb_avg = torch.mean(outputs.last_hidden_state, axis=0)
                embeddings.append(emb_avg)

        return embeddings


class ZeroscopeTextEmbeddingPipeline:
    def __init__(self, device="cuda"):
        # TODO fix this, we're loading the full zeroscope model incl UNet just to use the _encode_prompt function
        self.pipe = DiffusionPipeline.from_pretrained("../zeroscope_v2_576w")
        self.pipe = self.pipe.to(device)
        self.device = device

    def __call__(self, caption_batch):
        with torch.no_grad():
            embeddings = self.pipe._encode_prompt(
                caption_batch,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        return embeddings


class ZeroscopeImageEmbeddingPipeline:
    def __init__(self, device="cuda", caption_first=True):
        self.device = device
        self.caption_first = caption_first
        if self.caption_first:
            self.img_captioning_pipe = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-large",
                device=self.device,
            )
            self.text_emb_pipe = ZeroscopeTextEmbeddingPipeline(device)
        else:
            self.pipe = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

    def __call__(self, img_batch):
        if self.caption_first:
            caption_batch = self.img_captioning_pipe(img_batch)
            caption_batch = [caption[0]["generated_text"] for caption in caption_batch]
            embeddings = self.text_emb_pipe(caption_batch)
        else:
            with torch.no_grad():
                embeddings = self.pipe(
                    pixel_values=img_batch,
                    device=self.device,
                    return_tensors="pt",
                )
                embeddings = self.pipe.model.encode_image(embeddings.pixel_values)
        return embeddings


class ZeroscopeVideoEmbeddingPipeline:
    def __init__(self, device="cuda", caption_first=True):
        self.device = device
        self.caption_first = caption_first
        if self.caption_first:
            self.vid_captioning_pipe = pipeline(
                "image-to-text",
                model="kpyu/video-blip-flan-t5-xl-ego4d",
                device=self.device,
            )
            self.text_emb_pipe = ZeroscopeTextEmbeddingPipeline(device)
        else:
            self.pipe = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

    def __call__(self, vid_batch):
        if self.caption_first:
            # Use the first frame of the video to get caption (TODO improve)
            vid_batch = [Image.fromarray(frames[0].numpy()) for frames in vid_batch]
            caption_batch = self.vid_captioning_pipe(vid_batch)
            caption_batch = [caption[0]["generated_text"] for caption in caption_batch]

            print("caption_batch", caption_batch)
            print("caption_batch.shape", len(caption_batch))
            print("caption_batch type", type(caption_batch))

            embeddings = self.text_emb_pipe(caption_batch)
            print("embeddings.shape", embeddings.shape)
        else:
            with torch.no_grad():
                embeddings = self.pipe(
                    text=vid_batch,
                    device=self.device,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                embeddings = self.pipe.model.encode_text(
                    embeddings.input_ids, embeddings.attention_mask
                )
        return embeddings


class SVDVideoEmbeddingPipeline:
    def __init__(self, device="cuda"):
        self.device = device

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.enable_model_cpu_offload()

    def __call__(self, vid_batch):
        with torch.no_grad():
            embeddings = self.pipe._encode_image(
                vid_batch,
                device=self.device,
                num_videos_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            # embeddings are [N, 1, 1024]
            embeddings = embeddings[:, 0]

        return embeddings


class SVDVideoEmbeddingAndLatentsPipeline:
    def __init__(self, device="cuda"):
        self.device = device

        self.pipe: CustomStableVideoDiffusionPipeline = (
            CustomStableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
                torch_dtype=torch.float16,
                variant="fp16",
            )
        )
        self.pipe.enable_model_cpu_offload()

    def __call__(self, vid_batch):
        with torch.no_grad():
            embeddings_and_latents = self.pipe.get_embeddings_and_latents(vid_batch)
            packed = self.pipe.pack_embeddings_and_latents(**embeddings_and_latents)

        return packed


def save_embedding(path, embedding):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.save(path, embedding.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset to extract embeddings from. One of bmd, bmd_captions, had, nsd, nod, cc2017",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default=None,
        help="Path to the stimuli. Can be folder of frame_folder or folder of videos.",
    )
    parser.add_argument(
        "-m", "--metadata_path", type=str, default=None, help="Path to the metadata."
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default=None,
        help="Path to save the extracted embeddings.",
    )
    parser.add_argument(
        "-c",
        "--caption_first",
        type=bool,
        default=True,
        help="Whether to caption the image/video first, then send that through the text_encoder to get the final embedding",
    )
    parser.add_argument(
        "-f",
        "--load_from_frames",
        type=bool,
        default=False,
        help="Whether to load the video from frames (True) or from mp4 (False).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on."
    )
    parser.add_argument(
        "-e",
        "--emb_wanted",
        type=str,
        default="zeroscope_c",
        help="Which embedding to extract. One of zeroscope_c, clip_c",
    )

    args = parser.parse_args()

    main_alt(args)
