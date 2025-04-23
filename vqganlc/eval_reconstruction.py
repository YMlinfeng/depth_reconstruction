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
from torch.utils.data import Dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from scipy.stats import entropy
from models.models_vq import VQModel
import util.misc as misc
from PIL import Image
import yaml
import torch
from omegaconf import OmegaConf
import importlib
from torchvision.models.inception import inception_v3
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from models.models_vq import VQModel
from omegaconf import OmegaConf

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_config(config_path, display=True):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR")
    parser.add_argument("--output_dir", default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/vqganlc/output_dir", help="path where to save, empty for no saving")
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

    parser.add_argument("--n_vision_words", default=100000, type=int)
    parser.add_argument("--n_class", default=100, type=int)
    parser.add_argument("--disc_start", default=10000, type=int)
    parser.add_argument("--rate_q", type=float, default=1, help="Decoding Loss")
    parser.add_argument("--rate_p", type=float, default=1, help="VGG Loss")
    parser.add_argument("--vq_config_path", type=str, default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/vqganlc/vqgan_configs/vq-f16.yaml", help="Decoding Loss")
    parser.add_argument("--image_size", type=int, default=256, help="Decoding Loss")
    parser.add_argument("--tuning_codebook", type=int, default=1, help="Decoding Loss")
    parser.add_argument("--stage", type=int, default=1, help="Decoding Loss")
    parser.add_argument("--stage_1_ckpt", type=str, default="", help="Decoding Loss")
    parser.add_argument("--embed_dim", type=int, default=8, help="Decoding Loss")
    parser.add_argument("--quantizer_type", type=str, default="org", help="Decoding Loss")
    parser.add_argument("--rate_d", type=float, default=1, help="GAN Loss")
    parser.add_argument("--use_cblinear", type=int, default=2, help="Decoding Loss")
    parser.add_argument("--local_embedding_path", default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/vqganlc/codebook-100K.pth", type=str)
    parser.add_argument("--dataset", type=str, default="ffhq", help="")
    return parser


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    config = load_config(args.vq_config_path, display=True)

    model = VQModel(args=args, **config.model.params)
    model.to(device)
    model.eval()
    
    recons_save_dir = os.path.join(args.output_dir, "recons")
    os.makedirs(recons_save_dir, exist_ok=True)
    
    count = 0

    # # ---新编代码---
    # 读取输入图像
    image_path = "/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/vqganlc/dog.jpeg"
    image = Image.open(image_path).convert("RGB")

    # 预处理：调整大小、归一化、转换为 Tensor
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # 调整到 256×256
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

    # 伪造 DataLoader 结构，保持 for-loop 逻辑
    data_loader = [[image_tensor, image_tensor, torch.tensor([0]).to(device)]]
    for data_iter_step, (images, clip_image, label_cls) in enumerate(data_loader):

        # 处理输入数据
        b = images.shape[0]
        x = images.to(device)
        clip_image = clip_image.to(device)
        label_cls = label_cls.to(device)

        # VQ-GAN 处理
        with torch.no_grad():
            _, xrec = model(x, clip_image, data_iter_step, step=0, is_val=True)

        # 反归一化
        xrec[xrec > 1] = 1
        xrec[xrec < -1] = -1
        save_xrec = (xrec + 1) * 127.5
        save_xrec = save_xrec.squeeze(0).cpu().numpy()
        save_xrec = np.clip(save_xrec, 0, 255).astype(np.uint8)
        save_xrec = np.transpose(save_xrec, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        # 保存重建图像
        output_path = os.path.join(recons_save_dir, f"{count}.png")
        Image.fromarray(save_xrec).save(output_path)
        print(f"重建图像已保存：{output_path}")

        count += 1

    # 可视化原始图像与重建图像
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(save_xrec)
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")

    plt.show()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
