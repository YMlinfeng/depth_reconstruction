import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        return out


# 导入必要的模块（假设其他依赖模块如cv2，torch.nn等已导入）
import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设DINOv2, DPTHead, Compose, Resize, NormalizeImage, PrepareForNet等也已经导入或定义

# 定义一个深度估计模型，继承自nn.Module（PyTorch模型的基类）
class DepthAnythingV2(nn.Module):  # 此类用于深度估计，结合预训练Transformer编码器和特定的深度预测头
    def __init__(
        self,
        encoder="vitl",              # 编码器名称，默认"vitl"（Vision Transformer Large）
        features=256,                # depth_head部分的内部特征维数
        out_channels=[256, 512, 1024, 1024],  # 输出各层的通道数
        use_bn=False,                # 是否使用Batch Normalization（BN，一种归一化方法）
        use_clstoken=False,          # 是否使用类别Token；Transformer里有时会在序列开始加上一个类别token来捕捉全局信息
    ):
        super(DepthAnythingV2, self).__init__()  # 初始化父类
        
        # 定义不同编码器对应的中间层索引，每个编码器名称对应需要提取的Transformer层的索引
        self.intermediate_layer_idx = {  
            "vits": [2, 5, 8, 11],  # "vits"对应索引，列表中数字表示第几层
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        
        self.encoder = encoder   # 保存编码器名称
        # 初始化一个预训练的DINOv2模型，DINOv2是一种自监督预训练的视觉Transformer模型
        self.pretrained = DINOv2(model_name=encoder)  # pretrained 是一个预训练编码器，负责将图片转换成特征
        
        # 用预训练模型的嵌入维度作为深度预测头输入的维度
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,   # 嵌入维度，例如：如果vitl，可能为1024或其它数值
            features,                    # 内部特征维度256
            use_bn,                      # 是否使用BN
            out_channels=out_channels,   # 输出通道配置列表
            use_clstoken=use_clstoken,   # 是否包含类别token的信息
        )
        
    # __init__方法宏观解释：
    # 这部分代码的主要目的是初始化模型，设置预训练的编码器，以及根据预训练模型的输出特征，
    # 构造一个深度预测头（DPTHead)来实现深度估计。涉及的专有名词包括Transformer（视觉Transformer）、Batch Normalization（BN）和类别Token。

    def forward(self, x, return_depth=False, return_class_token=True):
        # 输入变量：
        # x: 图像张量， shape = [batch_size, channels, height, width]
        # 根据已知信息，x的一个例子形状是 torch.Size([3, 3, 518, 784])
        #   - batch_size=3（即一次输入3张图片）
        #   - channels=3（即RGB三通道颜色）
        #   - height=518, width=784（图片的高和宽）
        
        # 计算补丁尺寸：Transformer对图像通常会进行分块处理，patch_h, patch_w是分块后图像的尺寸
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14  # 整数除法计算中间patch的高和宽
        
        # 从预训练模型中提取中间层的特征
        features = self.pretrained.get_intermediate_layers(
            x,  # 输入的图像张量
            self.intermediate_layer_idx[self.encoder],  # 根据编码器名称获取需要提取的层索引
            return_class_token=return_class_token,  # 是否返回类别token，一般用于捕捉全局信息
        )
        # 变量说明：
        # features可能是一个列表，每个元素是某个中间层的输出特征，数据类型为Torch Tensor。
        # 每个Tensor的shape一般为 [batch_size, num_tokens, feature_dim] 或可能包含图像空间维度
        
        if return_depth:
            # 使用深度预测头计算深度图，基于预训练模型提取的特征以及patch尺寸（patch_h, patch_w)
            depth = self.depth_head(features, patch_h, patch_w)  # depth: Tensor，shape可能为 [batch_size, 1, new_h, new_w]
            depth = F.relu(depth)  # relu函数将负值截断为0，保持Tensor非负（激活函数）
            # squeeze(1)将深度Tensor第1维（通道维度）压缩掉（如果通道为1），结果shape变为 [batch_size, new_h, new_w]
            return depth.squeeze(1), features  # 返回计算后的深度图和提取的特征
        else:
            return features  # 如果不需要返回深度图，则只返回特征

    # forward方法宏观解释：
    # 这一部分实现模型的前向传播。输入图像经过预训练Transformer提取中间层的特征后，
    # 通过深度预测头（DPTHead）得到深度图。当参数return_depth为True时返回深度图（并经过relu激活），否则只返回特征。
    # 这里涉及的数据类型为Torch Tensor，shape操作有助于将原始图像映射为合适的分块尺寸。

    @torch.no_grad()  # 装饰器表示该方法在运行时不需要计算梯度，加快推理速度并降低显存占用
    def infer_image(self, raw_image, input_size=518):
        # raw_image: 原始图像数据（例如：使用OpenCV读入的numpy数组），shape通常为 [height, width, channels]
        # input_size: 希望输入到模型中的图像尺寸（长边/短边被调整到该值），默认518
        
        # 将原始图像转换为Tensor，并进行必要的预处理
        image, (h, w) = self.image2tensor(raw_image, input_size)
        # image: 转化后的图像Tensor，shape为 [1, channels, new_height, new_width]，1为批次数
        # (h, w): 原始图像的高和宽，用于后续将深度图恢复成原始尺寸
        
        # 利用前向传播得到深度图和特征
        depth, features = self.forward(image)
        # depth: 通常为 [batch_size, new_h, new_w]；这里batch_size为1
        
        # 利用F.interpolate将深度图调整至原始图像尺寸，[h, w]
        depth = F.interpolate(
            depth[:, None],  # 添加一个channel维度以符合插值函数的输入要求，shape变为 [batch_size, 1, new_h, new_w]
            (h, w),         # 目标尺寸：原始图像大小 (h, w)
            mode="bilinear",         # 双线性插值，适合连续变量（深度图）
            align_corners=True       # 对齐角点的插值模式
        )[0, 0]  # 索引取出batch和channel两个为1的维度，最后得到 [h, w] 的深度图Tensor
        
        return depth.cpu().numpy()  # 将Tensor放回CPU并转为numpy数组，便于后续处理或展示

    # infer_image方法宏观解释：
    # 该方法主要用于推理阶段，将原始图像进行预处理转换为Tensor，
    # 然后调用forward函数计算深度图，再通过双线性插值恢复为原始图像大小，最后返回numpy数组形式的深度图。
    # 这里使用@torch.no_grad()表示在推理过程中不需要计算梯度，从而节省资源。

    def image2tensor(self, raw_image, input_size=518):
        # 定义图像预处理管道，此处使用Compose将多个预处理步骤组合在一起
        transform = Compose(
            [
                Resize(
                    width=input_size,         # 将图像宽度调整到input_size（但此处会保持长宽比例）
                    height=input_size,        # 将图像高度调整到input_size
                    resize_target=False,      # 不调整标签（针对监督任务时的处理，本例不使用）
                    keep_aspect_ratio=True,   # 保持原图长宽比
                    ensure_multiple_of=14,    # 保证尺寸是14的倍数（因为Transformer分patch时通常要求尺寸能整除patch大小）
                    resize_method="lower_bound",  # 调整方法，这里采用“下界”模式
                    image_interpolation_method=cv2.INTER_CUBIC,  # 使用三次插值法进行图像插值
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化图像，常见的均值和标准差设置用于ImageNet预训练模型
                PrepareForNet(),  # 将数据格式转成网络可以接受的格式（例如：改变通道顺序）
            ]
        )
        # 变量说明：
        # transform 是一个组合流水线，将在接下来的代码中用于预处理raw_image

        # 获取原始图像尺寸，raw_image是numpy数组，shape格式为 [height, width, channels]
        h, w = raw_image.shape[:2]  # 取前两个值，即原始高和宽
        
        # 将图像颜色从BGR转换为RGB（OpenCV默认使用BGR格式），并归一化到[0, 1]范围
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0  # image类型为numpy.ndarray，shape= [h, w, channels]
        
        # 将图像通过预处理流水线转换
        image = transform({"image": image})["image"]  # 预处理后返回的字典中取出"image"的数据
        # image此时可能为一个numpy数组，形状为 [new_height, new_width, channels] 或已有通道转换（PrepareForNet通常会将通道移到前面）
        
        # 转换为PyTorch的Tensor，并增加一个batch维度（unsqueeze(0)使得batch_size=1）
        image = torch.from_numpy(image).unsqueeze(0)  # image类型为Tensor，shape = [1, channels, new_height, new_width]
        
        # 自动选择运行设备：优先使用CUDA，其次mps（Apple silicon），否则使用cpu
        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        image = image.to(DEVICE)  # 将image数据发送到选定的计算设备
        
        return image, (h, w)  # 返回预处理后的图像Tensor和原始图像的尺寸 (h, w)

    # image2tensor方法宏观解释：
    # 该方法将输入的原始numpy图像数据转换为适用于PyTorch网络的张量，并执行一系列预处理操作：
    # 颜色空间转换、归一化、尺寸调整以满足Transformer输入要求，再增加batch维度和迁移到相应的设备。
    # 专有名词包括Tensor、归一化（Normalize）、批尺寸（batch dimension）以及设备（DEVICE）。
