#!/usr/bin/env python3
import argparse
import os
import json
import cv2
import imageio
import torch
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms

def save_frames_to_folder(video_file, frames, is_bgr=True):
    """
    保存视频的每一帧到一个与视频同名的文件夹中。
    
    参数:
      video_file (str): 视频文件的路径。
      frames (list 或 numpy 数组): 要保存的帧序列。
      is_bgr (bool): 如果为 True，则先将帧从 BGR 转换为 RGB 后再保存（适用于 cv2.imread 读取的帧）。
    """
    folder = os.path.splitext(video_file)[0]
    os.makedirs(folder, exist_ok=True)
    for idx, frame in enumerate(frames):
        if is_bgr:
            save_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            save_frame = frame
        frame_filename = os.path.join(folder, f"frame_{idx:03d}.png")
        imageio.imwrite(frame_filename, save_frame)

# ---------------------------------------------------------
# 生成类别1视频：依次读取 total_frames 帧，每帧取出三个视角分别写入独立视频
# ---------------------------------------------------------
def create_category1_videos(jsonl_file, output_dir, total_frames=34, fps=8, resize_width=None, resize_height=None):
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < total_frames:
        raise ValueError(f"jsonl 文件不足 {total_frames} 帧")

    front_frames = []
    frontleft_frames = []
    frontright_frames = []
    
    for i in range(total_frames):
        sample = json.loads(lines[i])
        # 分别读取三个视角的图片路径
        front_img = cv2.imread(sample['images']['CAM_FRONT'])
        frontleft_img = cv2.imread(sample['images']['CAM_FRONT_LEFT'])
        frontright_img = cv2.imread(sample['images']['CAM_FRONT_RIGHT'])
        if front_img is None or frontleft_img is None or frontright_img is None:
            raise ValueError(f"在第 {i+1} 帧中读取图片失败，请检查图片路径!")
        # 如果提供了 resize 参数，则统一调整图像尺寸
        if resize_width is not None and resize_height is not None:
            front_img = cv2.resize(front_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            frontleft_img = cv2.resize(frontleft_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            frontright_img = cv2.resize(frontright_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            
        front_frames.append(front_img)
        frontleft_frames.append(frontleft_img)
        frontright_frames.append(frontright_img)

    # 定义输出文件路径
    front_video_file = os.path.join(output_dir, "front.mp4")
    frontleft_video_file = os.path.join(output_dir, "frontleft.mp4")
    frontright_video_file = os.path.join(output_dir, "frontright.mp4")

    # 写视频时，imageio 要求图片为 RGB 格式，所以先转换
    writer_front = imageio.get_writer(front_video_file, fps=fps, codec='libx264')
    for frame in front_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer_front.append_data(frame_rgb)
    writer_front.close()
    print(f"生成视频: {front_video_file}")
    # 同时保存每一帧到与视频同名的文件夹中
    save_frames_to_folder(front_video_file, front_frames, is_bgr=True)

    writer_frontleft = imageio.get_writer(frontleft_video_file, fps=fps, codec='libx264')
    for frame in frontleft_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer_frontleft.append_data(frame_rgb)
    writer_frontleft.close()
    print(f"生成视频: {frontleft_video_file}")
    save_frames_to_folder(frontleft_video_file, frontleft_frames, is_bgr=True)

    writer_frontright = imageio.get_writer(frontright_video_file, fps=fps, codec='libx264')
    for frame in frontright_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer_frontright.append_data(frame_rgb)
    writer_frontright.close()
    print(f"生成视频: {frontright_video_file}")
    save_frames_to_folder(frontright_video_file, frontright_frames, is_bgr=True)


# ---------------------------------------------------------
# 生成类别2视频：取出一帧图片的三个视角，按照左中右循环构造 total_frames 帧
# ---------------------------------------------------------
def create_category2_video(jsonl_file, output_dir, total_frames=34, fps=8, resize_width=None, resize_height=None):
    with open(jsonl_file, 'r') as f:
        line = f.readline()
    sample = json.loads(line)
    left_img = cv2.imread(sample['images']['CAM_FRONT_LEFT'])
    center_img = cv2.imread(sample['images']['CAM_FRONT'])
    right_img = cv2.imread(sample['images']['CAM_FRONT_RIGHT'])
    if left_img is None or center_img is None or right_img is None:
        raise ValueError("读取第一帧图片失败，请检查图片路径！")
    # 如果提供了 resize 参数，则调整尺寸
    if resize_width is not None and resize_height is not None:
        left_img = cv2.resize(left_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        center_img = cv2.resize(center_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    # 按照 左中右 顺序排列
    pattern = [left_img, center_img, right_img]

    frames = []
    for i in range(total_frames):
        frames.append(pattern[i % len(pattern)])
    video_file = os.path.join(output_dir, "111.mp4")
    writer = imageio.get_writer(video_file, fps=fps, codec='libx264')
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
    writer.close()
    print(f"生成视频: {video_file}")
    save_frames_to_folder(video_file, frames, is_bgr=True)

# ---------------------------------------------------------
# 生成类别3视频：依次取时间步0、1、2的左、中、右三个视角图片，总共9个画面
# ---------------------------------------------------------
def create_category3_video(jsonl_file, output_dir, fps=2, resize_width=None, resize_height=None):
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()

    # 至少需要3个时间步（3行数据）
    if len(lines) < 3:
        raise ValueError("jsonl 文件不足 3 帧，无法生成类别3视频")

    frames = []
    # 遍历时间步 0 到 2，每个时间步取左、中、右三帧
    for t in range(3):
        sample = json.loads(lines[t])
        # 分别读取左摄像头、中摄像头和右摄像头的图片
        left_img = cv2.imread(sample['images']['CAM_FRONT_LEFT'])
        center_img = cv2.imread(sample['images']['CAM_FRONT'])
        right_img = cv2.imread(sample['images']['CAM_FRONT_RIGHT'])
        # 获取图片尺寸
        # print(f"Left Image Size: {left_img.shape}")    # (高, 宽, 通道数)
        # print(f"Center Image Size: {center_img.shape}")
        # print(f"Right Image Size: {right_img.shape}")
        if left_img is None or center_img is None or right_img is None:
            raise ValueError(f"在时间步 {t} 的帧中读取图片失败，请检查图片路径!")

        # 如果提供了 resize 参数，则统一调整图像尺寸
        if resize_width is not None and resize_height is not None:
            left_img = cv2.resize(left_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            center_img = cv2.resize(center_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

        # 按照左、中、右顺序加入 frames 列表
        frames.append(left_img)
        frames.append(center_img)
        frames.append(right_img)

    # 定义输出文件路径
    video_file = os.path.join(output_dir, "category3.mp4")
    writer = imageio.get_writer(video_file, fps=fps, codec='libx264')
    
    # 写视频时，转换 BGR 到 RGB 格式
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
    writer.close()
    print(f"生成视频: {video_file}")
    save_frames_to_folder(video_file, frames, is_bgr=True)

# ---------------------------------------------------------
# 以下函数为 CogVideoX 的编码/解码推理代码
# ---------------------------------------------------------
def encode_video(model_path, ckpt_path, video_path, dtype, device, resize_width=None, resize_height=None):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and encodes the video frames.
    """
    # model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model = load_model_from_pth(
        config_path="/mnt/bn/occupancy3d/workspace/mzj/CogVideoX-2b/vae/config.json",
        ckpt_path=ckpt_path,
        dtype=dtype, 
        device=device
    )
    model.enable_slicing()
    model.enable_tiling()

    video_reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []
    for frame in video_reader:
        # 对读取的视频帧进行 resize，确保输入宽高一致
        if resize_width is not None and resize_height is not None:
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        frames.append(transforms.ToTensor()(frame))
    video_reader.close()
    # frames_tensor shape: (channels, frames, height, width)
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

    with torch.no_grad():
        encoded_frames = model.encode(frames_tensor)[0].sample()
    return encoded_frames


def decode_video(model_path, ckpt_path, encoded_tensor_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and decodes the encoded video frames.
    """
    # model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model = load_model_from_pth(
        config_path="/mnt/bn/occupancy3d/workspace/mzj/CogVideoX-2b/vae/config.json",
        ckpt_path=ckpt_path,
        dtype=dtype, 
        device=device
    )
    encoded_frames = torch.load(encoded_tensor_path, map_location=device)
    encoded_frames = encoded_frames.to(device).to(dtype)
    with torch.no_grad():
        decoded_frames = model.decode(encoded_frames).sample
    return decoded_frames


def save_video(tensor, output_path, fps):
    """
    Saves the video frames to a video file with user-defined FPS and also saves individual frames as images.
    """
    tensor = tensor.to(dtype=torch.float32)
    # tensor shape expected: (1, channels, frames, height, width)
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)
    video_out = os.path.join(output_path, "output.mp4")
    writer = imageio.get_writer(video_out, fps=fps, codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"推理输出视频保存至: {video_out}")
    # 此处视频的帧经过模型恢复后应为 RGB 格式，因此不需转换
    save_frames_to_folder(video_out, frames, is_bgr=False)

def load_model_from_pth(config_path, ckpt_path, dtype, device):
    import json
    with open(config_path, "r") as f:
        config = json.load(f)

    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)

    model = AutoencoderKLCogVideoX(**config).to(device).to(dtype)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "vqvae_state_dict" in checkpoint:
        state_dict = checkpoint["vqvae_state_dict"]
    else:
        raise ValueError("Checkpoint does not contain 'vqvae_state_dict' key.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    model.eval()

    return model
# ---------------------------------------------------------
# 主函数：对不同模式进行处理
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo with video stitching from JSONL data")
    parser.add_argument("--ckpt_path", type=str, default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints_vae_finetune/vqvae_epoch1_step50.pth",
                        help="The path to the CogVideoX model")
    parser.add_argument("--model_path", type=str, default="/mnt/bn/occupancy3d/workspace/mzj/CogVideoX-2b/vae/",
                        help="The path to the CogVideoX model")
    parser.add_argument("--video_path", type=str, default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/ooout/category3.mp4",
                        help="The path to the video file (for encoding). 如果同时提供了 --jsonl_file 则生成视频后可通过 --video_choice 指定使用哪个视频")
    parser.add_argument("--encoded_path", type=str, default="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/ooout/mzj_test_encoded.pt",
                        help="The path to the encoded tensor file (for decoding)")
    parser.add_argument("--output_path", type=str, default="ooout",
                        help="The path to save the output file and generated videos")
    parser.add_argument("--mode", type=str, choices=["encode", "decode", "both"], default="both",
                        help="Mode: encode, decode, or both")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="The data type for computation (e.g., 'float16' or 'bfloat16')")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to use for computation (e.g., 'cuda' or 'cpu')")
    # 新增参数：JSONL文件路径，用于视频生成
    parser.add_argument("--jsonl_file", type=str,
                        default="/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec.jsonl")
    # 新增参数：生成视频选择（可选："front", "frontleft", "frontright", "111"），默认 "front"
    parser.add_argument("--video_choice", type=str, choices=["front", "frontleft", "frontright", "111"], default="front",
                        help="If --jsonl_file is provided, select which generated video to use for inference")
    # 新增参数：输入视频帧的宽与高
    parser.add_argument("--input_width", type=int, default=450, help="输入视频帧的宽度")
    parser.add_argument("--input_height", type=int, default=300, help="输入视频帧的高度")
    # 新增参数：FPS
    parser.add_argument("--fps", type=int, default=2, help="视频的帧率设置")

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # 如果提供了 jsonl 文件路径，则先生成视频，并对每帧数据进行 resize
    if args.jsonl_file:
        # print("开始生成第一类视频（front.mp4, frontleft.mp4, frontright.mp4）...")
        # create_category1_videos(
        #     args.jsonl_file, 
        #     args.output_path, 
        #     total_frames=17, 
        #     fps=args.fps, 
        #     resize_width=args.input_width, 
        #     resize_height=args.input_height
        # )
        # print("开始生成第二类视频（111.mp4）...")
        # create_category2_video(
        #     args.jsonl_file, 
        #     args.output_path, 
        #     total_frames=17, 
        #     fps=args.fps, 
        #     resize_width=args.input_width, 
        #     resize_height=args.input_height
        # )
        print("开始生成第三类视频（category3.mp4）...")
        create_category3_video(
            args.jsonl_file,
            args.output_path,
            fps=args.fps,
            resize_width=args.input_width,
            resize_height=args.input_height
        )
        # 如果没有直接提供 --video_path，则按 --video_choice 参数选择生成的视频
        if not args.video_path:
            args.video_path = os.path.join(args.output_path, f"{args.video_choice}.mp4")
            print(f"未提供 --video_path，自动选择生成的视频: {args.video_path}")

    # 根据不同模式进行编码/解码推理
    if args.mode == "encode":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(
            args.model_path, 
            args.video_path, 
            dtype, 
            device, 
            resize_width=args.input_width, 
            resize_height=args.input_height
        )
        tensor_save_path = os.path.join(args.output_path, "mzj_test_encoded.pt")
        torch.save(encoded_output, tensor_save_path)
        print(f"Finished encoding the video. Encoded tensor saved at {tensor_save_path}")

    elif args.mode == "decode":
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        decoded_output = decode_video(args.model_path, args.encoded_path, dtype, device)
        save_video(decoded_output, args.output_path, fps=args.fps)
        print(f"Finished decoding the video and saved it to {args.output_path}/output.mp4")

    elif args.mode == "both":
        assert args.video_path, "Video path must be provided for encoding in 'both' mode."
        encoded_output = encode_video(
            args.model_path, 
            args.ckpt_path,
            args.video_path, 
            dtype, 
            device, 
            resize_width=args.input_width, 
            resize_height=args.input_height,
        )
        encoded_tensor_path = os.path.join(args.output_path, "encoded.pt")
        torch.save(encoded_output, encoded_tensor_path)
        decoded_output = decode_video(args.model_path, args.ckpt_path, encoded_tensor_path, dtype, device)
        save_video(decoded_output, args.output_path, fps=args.fps)
        print(f"Finished encoding and decoding the video in 'both' mode.")