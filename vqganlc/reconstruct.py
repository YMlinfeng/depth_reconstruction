import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path
import albumentations
from scipy import linalg
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from scipy.stats import entropy

from models.models_vq import VQModel
import util.misc as misc

from PIL import Image
import yaml
import torch
from omegaconf import OmegaConf
import importlib
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np

import matplotlib.pyplot as plt
import torch
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)

    # Dataset parameters
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")


    parser.add_argument("--imagenet_path", default="", type=str, help="path of llama model")

    parser.add_argument("--n_vision_words", default=32000, type=int)
    parser.add_argument("--n_class", default=100, type=int)
    parser.add_argument("--disc_start", default=10000, type=int)
    parser.add_argument("--rate_q", type=float, default=1, help="Decoding Loss")
    parser.add_argument("--rate_p", type=float, default=1, help="VGG Loss")

    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/model_16384.yaml", help="Decoding Loss")
    parser.add_argument("--image_size", type=int, default=256, help="Decoding Loss")
    parser.add_argument("--tuning_codebook", type=int, default=1, help="Decoding Loss")
    parser.add_argument("--stage", type=int, default=1, help="Decoding Loss")

    parser.add_argument("--stage_1_ckpt", type=str, default="", help="Decoding Loss")
    parser.add_argument("--embed_dim", type=int, default=8, help="Decoding Loss")
    parser.add_argument("--quantizer_type", type=str, default="org", help="Decoding Loss")
    parser.add_argument("--rate_d", type=float, default=1, help="GAN Loss")

    parser.add_argument("--use_cblinear", type=int, default=0, help="Decoding Loss")
    parser.add_argument("--local_embedding_path", default="cluster_codebook_1000cls_100000.pth", type=str)
    
    parser.add_argument("--dataset", type=str, default="ffhq", help="")

    return parser


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


def get_preprocessor(args):

    #rescaler = albumentations.SmallestMaxSize(max_size=args.image_size)
    #cropper = albumentations.CenterCrop(height=args.image_size, width=args.image_size)
    rescaler = albumentations.LongestMaxSize(max_size=args.image_size)
    cropper = albumentations.PadIfNeeded(
        min_height=args.image_size, min_width=args.image_size,
        position="top_left", border_mode=cv2.BORDER_CONSTANT, fill=0)
    preprocessor = albumentations.Compose([rescaler, cropper])
    return preprocessor

def get_model(args):
    device = torch.device(args.device)
    config = load_config(args.vq_config_path, display=True)
    model = VQModel(args=args, **config.model.params)
    model.to(device)
    if "last" in args.stage_1_ckpt:
        sd = torch.load(os.path.join(args.stage_1_ckpt), map_location="cpu")["model"]
    else:
        sd = torch.load(os.path.join(args.stage_1_ckpt), map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(missing, unexpected)
    model.eval()
    return model


def main(args):
    recons_save_dir = os.path.join(args.output_dir, "recons")
    os.makedirs(recons_save_dir, exist_ok=True)
    model = get_model(args)

    preprocessor = get_preprocessor(args)
    image = Image.open(args.imagenet_path).convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = preprocessor(image=image)["image"]
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = image.transpose(2, 0, 1)# chw
    images = image[None, :, :, :]
    x = torch.Tensor(images).to(args.device)

    count = 0
    with torch.no_grad():
        _, _, _, _, _, tk_labels, xrec = model(x, None, 0, step=0, is_val=True)
        #import pdb; pdb.set_trace()

    save_x = (x + 1) * 127.5  
    xrec[xrec > 1] = 1
    xrec[xrec < -1] = -1
    save_xrec = (xrec + 1) * 127.5

    ####
    save_x = save_x / 255.0
    save_xrec = save_xrec / 255.0

    for b in range(0, save_x.shape[0]):
        plt.imsave(os.path.join(recons_save_dir, "%s_origin.png"%(count)), np.uint8(save_x[b].detach().cpu().numpy().transpose(1, 2, 0) * 255))
        plt.imsave(os.path.join(recons_save_dir, "%s.png"%(count)), np.uint8(save_xrec[b].detach().cpu().numpy().transpose(1, 2, 0) * 255))
    count = count + 1




if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

