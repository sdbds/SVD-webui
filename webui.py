import argparse
import os
import sys
from omegaconf import OmegaConf
import torch
import math
import cv2
import numpy as np
from einops import rearrange, repeat
import gradio as gr
import random

sys.path.append("generative-models")
from sgm.util import instantiate_from_config

# from sgm.inference.helpers import embed_watermark
# from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering

from glob import glob
from pathlib import Path
from typing import Optional

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    config.model.params.conditioner_config.params.emb_models[
        0
    ].params.open_clip_embedding_config.params.init_device = device
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    with torch.device(device):
        model = (
            instantiate_from_config(config.model)
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    # filter = DeepFloydDataFiltering(verbose=False, device=device)
    filter = ""
    return model  # , filter


def get_unique_embedder_keys(conditioner):
    return list(set([embedder.input_key for embedder in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device, dtype=None):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device, dtype=dtype),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def sample(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    resize_image: bool = False,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 1026,
    decoding_t: int = 2,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = "./outputs",
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError
    all_out_paths = []
    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            if resize_image and image.size != (1024, 576):
                print(f"Resizing {image.size} to (1024, 576)")
                image = TF.resize(TF.resize(image, 1024), (576, 1024))
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug
        # low vram mode
        model.conditioner.cpu()
        model.first_stage_model.cpu()
        torch.cuda.empty_cache()
        model.sampler.verbose = True

        with torch.no_grad():
            with torch.autocast(device):
                model.conditioner.to(device)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )
                model.conditioner.cpu()
                torch.cuda.empty_cache()

                # from here, dtype is fp16
                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
                for k in uc.keys():
                    uc[k] = uc[k].to(dtype=torch.float16)
                    c[k] = c[k].to(dtype=torch.float16)

                randn = torch.randn(shape, device=device, dtype=torch.float16)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(
                    device,
                )
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                for k in additional_model_inputs:
                    if isinstance(additional_model_inputs[k], torch.Tensor):
                        additional_model_inputs[k] = additional_model_inputs[k].to(
                            dtype=torch.float16
                        )

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                samples_z.to(dtype=model.first_stage_model.dtype)
                ##

                model.en_and_decode_n_samples_a_time = decoding_t
                model.first_stage_model.to(device)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                model.first_stage_model.cpu()
                torch.cuda.empty_cache()

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    fps_id + 1,
                    (samples.shape[-1], samples.shape[-2]),
                )

                # samples = embed_watermark(samples)
                # samples = filter(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                for frame in vid:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)
                writer.release()
                all_out_paths.append(video_path)
    return all_out_paths


def infer(
    input_path: str,
    resize_image: bool,
    n_frames: int,
    n_steps: int,
    seed: str,
    decoding_t: int,
) -> str:
    if seed in {"random", "-1"}:
        seed = random.randint(0, 2**32)
    seed = int(seed)
    output_paths = sample(
        input_path=input_path,
        resize_image=resize_image,
        num_frames=n_frames,
        num_steps=n_steps,
        fps_id=6,
        motion_bucket_id=127,
        cond_aug=0.02,
        seed=1026,
        decoding_t=decoding_t,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        device=device,
        output_folder=args.outputs,
    )
    return output_paths[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="model_path(svd/svd_xt)")
    parser.add_argument("--outputs", help="directory to save segmentation image")
    parser.add_argument("--port", help="server port,default 7860", default=7860)
    args = parser.parse_args()

    # load model
    version = os.path.basename(args.model_path).split(".")[0]

    if version == "svd":
        num_frames = 14
        num_steps = 25
        model_config = "generative-models/scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = 25
        num_steps = 30
        model_config = "generative-models/scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = 14
        num_steps = 25
        model_config = (
            "generative-models/scripts/sampling/configs/svd_image_decoder.yaml"
        )
    elif version == "svd_xt_image_decoder":
        num_frames = 25
        num_steps = 30
        model_config = (
            "generative-models/scripts/sampling/configs/svd_xt_image_decoder.yaml"
        )
    else:
        raise ValueError(f"model {version} does not exist.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    # move models expect unet to cpu
    model.conditioner.cpu()
    model.first_stage_model.cpu()
    # change the dtype of unet
    model.model.to(dtype=torch.float16)
    torch.cuda.empty_cache()
    model = model.requires_grad_(False)

    with gr.Blocks() as demo:
        with gr.Column():
            image = gr.Image(label="input image", type="filepath")
            resize_image = gr.Checkbox(label="resize to optimal size", value=True)
            btn = gr.Button("Run")
            with gr.Accordion(label="Advanced options", open=False):
                n_frames = gr.Number(
                    precision=0, label="number of frames", value=num_frames
                )
                n_steps = gr.Number(
                    precision=0, label="number of steps", value=num_steps
                )
                seed = gr.Text(
                    value="random",
                    label="seed (integer or 'random')",
                )
                decoding_t = gr.Number(
                    precision=0, label="number of frames decoded at a time", value=2
                )
        with gr.Column():
            video_out = gr.Video(label="generated video")
        examples = [
            [
                "https://user-images.githubusercontent.com/8085926/285306818-37a63a1b-2faa-48ab-93fe-6dd9ee5d52b2.png"
            ]
        ]
        inputs = [image, resize_image, n_frames, n_steps, seed, decoding_t]
        outputs = [video_out]
        btn.click(infer, inputs=inputs, outputs=outputs)
        gr.Examples(examples=examples, inputs=inputs, outputs=outputs, fn=infer)
        demo.queue().launch(
            inbrowser=True, debug=True, show_error=True, server_port=int(args.port)
        )
