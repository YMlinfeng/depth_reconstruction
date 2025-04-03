import torch
import sys
import os

# 权重文件路径
file_path = "/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_2dvqvaelargefewfewchannels/epoch_1.pth"
file_path = "/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints/vqvae_epoch1_step12500.pth"

#! 1
import debugpy
# 监听端口
debugpy.listen(("127.0.0.1", 5678))
# 等待调试器连接（可选）
print("等待调试器连接...") #按F5
debugpy.wait_for_client()
# 根据权重文件名称生成日志文件名称（保存在当前文件夹下）
base_name = os.path.basename(file_path)
log_filename = os.path.splitext(base_name)[0] + ".log"

# 自定义 Logger 类，用于同时写入 stdout 和日志文件
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 重定向 sys.stdout，使所有 print 输出同时写入日志文件
sys.stdout = Logger(log_filename, sys.stdout)

print(f"\n🔍 正在解析文件: {file_path}\n")

try:
    # 加载文件，自动转换到 CPU 以避免 GPU 相关问题
    data = torch.load(file_path, map_location="cpu") # data是dict，dict_keys(['meta', 'state_dict', 'optimizer'])
    
    if isinstance(data, dict):
        print("📌 检测到字典类型数据，可能是 `state_dict` 或自定义数据")

        # 检查是否是 `state_dict`
        if "model" in data and isinstance(data["model"], dict):
            print("🔹 可能是完整的 checkpoint，包含 `state_dict`")
            data = data["model"]  # 进入 `state_dict`
        elif "state_dict" in data and isinstance(data["state_dict"], dict):
            print("🔹 可能是 `state_dict`，尝试解析 `state_dict`")
            data = data["state_dict"]

        print("\n📜 **解析内容:**\n")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"🔹 `{key}`: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            else:
                print(f"🔹 `{key}`: 类型={type(value)} (非 Tensor)")
                
    elif isinstance(data, torch.nn.Module):
        print("📌 该文件包含完整的 PyTorch 模型")
        print("\n📜 **模型结构:**\n")
        print(data)
    
    else:
        print(f"⚠️ 无法识别的数据类型: {type(data)}，请手动检查！")

except Exception as e:
    print(f"❌ 解析失败！错误信息: {e}")



#! 2
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# """
# 这个脚本用于加载一个 PyTorch 权重文件，并打印出其中的内容结构。
# 文件路径为：/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints/vqvae_epoch1_step12500.pth
# """

# import torch

# def print_structure(obj, indent=0):
#     """
#     递归打印对象的内部结构，主要针对字典和列表。
#     """
#     space = ' ' * indent
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             print(f"{space}{key}: ({type(value).__name__})")
#             # 如果 value 是字典或列表，则递归打印
#             if isinstance(value, dict) or isinstance(value, list):
#                 print_structure(value, indent + 4)
#     elif isinstance(obj, list):
#         for idx, value in enumerate(obj):
#             print(f"{space}[{idx}]: ({type(value).__name__})")
#             if isinstance(value, dict) or isinstance(value, list):
#                 print_structure(value, indent + 4)
#     else:
#         print(f"{space}{value} ({type(obj).__name__})")

# if __name__ == '__main__':

#     checkpoint_file = "/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints/vqvae_epoch1_step12500.pth"
#     checkpoint_file = "/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth"
#     try:
#         # 加载权重文件，使用 map_location='cpu' 避免 GPU 环境依赖问题
#         checkpoint = torch.load(checkpoint_file, map_location='cpu')
#         print("加载的checkpoint类型:", type(checkpoint))
        
#         # 如果 checkpoint 是字典，则打印其 key 值
#         if isinstance(checkpoint, dict):
#             print("\ncheckpoint中的keys:")
#             for key in checkpoint.keys():
#                 print("  ", key)
            
#             print("\n详细结构：")
#             print_structure(checkpoint)
#         else:
#             # 如果不是字典，直接打印内容
#             print("加载的checkpoint不是字典类型:")
#             print(checkpoint)
#     except Exception as e:
#         print("加载checkpoint时出错:", str(e))