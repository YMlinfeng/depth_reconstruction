import argparse
import datetime
import json
import os
import time
from pathlib import Path
import albumentations
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import yaml
import torch
from omegaconf import OmegaConf

from vqganlc.models.vqgan_lc import VQModel 
from engine_training_vqgan import train_one_epoch
import util.misc as misc

from util.misc import NativeScalerWithGradNormCount as NativeScaler


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):

  model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x




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

    # Model parameters
    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")

    # Optimizer parameters
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
    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR")
    # Dataset parameters
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
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
    parser.add_argument("--imagenet_path", default="", type=str)
    parser.add_argument("--n_vision_words", default=16384, type=int)
    parser.add_argument("--n_class", default=1000, type=int)    
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/model_16384.yaml", help="Decoding Loss")
    parser.add_argument("--image_size", type=int, default=256, help="Image Size")
    parser.add_argument("--stage", type=int, default=1, help="VQ/GPT")
    parser.add_argument("--quantizer_type", type=str, default="org", help="EMA/ORG")

    parser.add_argument("--embed_dim", type=int, default=768, help="Feature Dim")
    parser.add_argument("--tuning_codebook", type=int, default=-1, help="Frozen or Tuning Coebook")
    parser.add_argument("--use_cblinear", type=int, default=0, help="Using Projector")

    parser.add_argument("--local_embedding_path", default="cluster_codebook_1000cls_100000.pth")
    parser.add_argument("--disc_start", default=10000, type=int, help="GAN Loss Start")
    parser.add_argument("--rate_q", type=float, default=0.1, help="Quant Loss")
    parser.add_argument("--rate_p", type=float, default=1, help="VGG Loss")
    parser.add_argument("--rate_d", type=float, default=0.1, help="GAN Loss")

    parser.add_argument("--dataset", type=str, default="imagenet", help="")

    return parser


def main(args):
    config = load_config(args.vq_config_path, display=True)

    model = VQModel(args=args, **config.model.params)
    model.to(device)
    model_without_ddp = model
    

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        model_without_ddp = model.module

    #####Using Projector
    if args.use_cblinear != 0:
        opt_ae = torch.optim.Adam(list(model_without_ddp.encoder.parameters())+
                                list(model_without_ddp.decoder.parameters())+
                                list(model_without_ddp.tok_embeddings.parameters())+
                                list(model_without_ddp.quant_conv.parameters())+
                                list(model_without_ddp.codebook_projection.parameters()) + 
                                list(model_without_ddp.post_quant_conv.parameters()), lr=args.lr, betas=(0.5, 0.9), eps=1e-7)
    else:
        opt_ae = torch.optim.Adam(list(model_without_ddp.encoder.parameters())+
                                list(model_without_ddp.decoder.parameters())+
                                list(model_without_ddp.tok_embeddings.parameters())+
                                list(model_without_ddp.quant_conv.parameters())+
                                list(model_without_ddp.post_quant_conv.parameters()), lr=args.lr, betas=(0.5, 0.9), eps=1e-7)
    opt_dist = torch.optim.Adam(model_without_ddp.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9), eps=1e-7)

    loss_scaler_ae = NativeScaler()
    loss_scaler_disc = NativeScaler()


    optimizer = [opt_ae, opt_dist]
    loss_scaler = [loss_scaler_ae, loss_scaler_disc]

    num_val_images = len(dataset_val.image_ids)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )
        
        misc.save_model_last_vqgan_ganloss(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
        )
        
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model_vqgan(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch#,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

