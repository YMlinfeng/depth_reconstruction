import pickle
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 定义pkl文件的路径（请根据实际情况替换）
pkl_file_path = '/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_data_bottom_static_scaleddepth_240925_sub.pkl'

with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# 检查 metadata 部分
metadata = data.get('metadata')
print("metadata 的类型:", type(metadata))
# 如果 metadata 是一个字典，可以打印它的一些键和值：
if isinstance(metadata, dict):
    print("metadata 中的键:", list(metadata.keys()))
    # 打印其中一些键对应的值（如果数据量不大）
    for key in list(metadata.keys())[:5]:  # 只打印前5个键的内容
        print(f"键 {key} 对应的值:", metadata[key])
else:
    print("metadata 内容:", metadata)

# 检查 infos 部分
infos = data.get('infos')
print("\ninfos 的类型:", type(infos))
# 假设 infos 是列表或字典
if isinstance(infos, list):
    print("infos 列表的长度:", len(infos))
    # 打印第一个元素内容
    if len(infos) > 0:
        print("第一个样本项的内容:", infos[0])
elif isinstance(infos, dict):
    print("infos 中的键:", list(infos.keys()))
    # 打印部分键对应的值
    for key in list(infos.keys())[:5]:
        print(f"键 {key} 对应的值:", infos[key])
else:
    print("infos 内容:", infos)


class MyVQGANDataset(Dataset):
    '''
    这个pkl文件是一个dict，包含两项，我们主要关注第二项，键为infos，值是一个列表（list），列表长度为 1,253,048，说明这个文件中包含了 1253048 个数据样本的信息。
        列表中的每个元素都是一个字典，描述了单个样本的详细信息。
        第一个样本项中的内容包含：
            frame_idx 和 timestamp：表示数据帧的索引和时间戳。
            scene_token：表示当前数据所属的场景标识，可以用来分割或者区分不同场景下的数据。
            cams：存放各个摄像头的信息。比如 "CAM_FRONT"、"CAM_FRONT_LEFT"、"CAM_FRONT_RIGHT" 等，每个摄像头记录了：
                data_path：实际存储该摄像头图像的路径。
                cam_intrinsic：一个 numpy 数组，保存了摄像头的内参矩阵（决定摄像头成像的焦距、主点等信息）。
                cam_extrinsic：一个 numpy 数组，保存了摄像头的外参矩阵（即摄像头相对于世界或者车辆的位姿信息）。
                depth_path：对应深度图的文件路径（可能经过尺度缩放处理）。
    '''
    def __init__(self, pkl_file, root_dir):
        """
        参数：
        - pkl_file: pkl文件的路径
        - root_dir: 数据文件的根目录（如果图片路径是相对路径）
        """
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        self.metadata = data.get('metadata')  # 全局元数据，如版本号
        self.infos = data.get('infos')  # 样本信息列表
        self.root_dir = root_dir

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        # 取得第 idx 个样本信息字典
        sample_info = self.infos[idx]
        
        # 示例：这里取 "CAM_FRONT" 摄像头的图像和深度数据
        cam_info = sample_info['cams'].get('CAM_FRONT')
        if cam_info is None:
            raise ValueError(f"样本 {idx} 中没有 'CAM_FRONT' 摄像头数据")
        
        # 构造图像完整路径和深度图完整路径
        img_path = os.path.join(self.root_dir, cam_info['data_path'])
        depth_path = os.path.join(self.root_dir, cam_info['depth_path'])
        
        # 加载图像
        image = Image.open(img_path).convert("RGB")
        # 加载深度图（假设为灰度图，可以调整需求）
        depth_image = Image.open(depth_path).convert("L")
        
        # 转换为 numpy 数组或 tensor，根据需要可以进一步处理
        image = np.array(image)
        depth_image = np.array(depth_image)
        
        # 返回图像、深度图以及本样本的其它信息
        return image, depth_image, sample_info

# 示例使用
pkl_file_path = '/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_data_bottom_static_scaleddepth_240925_sub.pkl'
root_dir = '/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk'  # 替换为图片、深度图实际存放的根目录

dataset = MyVQGANDataset(pkl_file_path, root_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 迭代第一个批次数据，验证数据加载器
for i, (images, depths, infos) in enumerate(dataloader):
    print("Batch", i, "图像形状:", images.shape, "深度图形状:", depths.shape)
    # 这里只读取一个 batch 来测试，之后将数据送入 VQGAN 训练流程
    break
