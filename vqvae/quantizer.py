import torch  # 导入 PyTorch 库，用于张量操作和构建神经网络
import torch.nn as nn  # 导入神经网络模块，方便实现模型组件（例如嵌入层、卷积层等）
import torch.nn.functional as F  # 导入功能函数，比如激活函数等（该代码中暂时未使用）
import numpy as np  # 导入 NumPy，用于数值计算
from einops import rearrange  # 导入einops，用于灵活的张量重排

# =======================================================================
# 模块名称：VectorQuantizer
# 目的：实现向量量化（Vector Quantization），将连续的特征向量映射到离散的嵌入空间中。
# 向量量化在一些VQ-VAE（向量量化变分自编码器）或离散潜变量模型中非常常见。
# 此外，该版本的量化器避免了昂贵的矩阵乘法，并可以后处理重映射索引。
# =======================================================================
class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: 由于一个 BUG，beta 项被应用到错误的项上；为了向后兼容，默认使用 BUG 版本，
    #      但可以通过设置 legacy=False 来修正。

    # -------------------------------------------------------------------
    # 构造函数 __init__
    # 目的：初始化 VectorQuantizer 对象，将传入的参数进行记录和组件的注册
    # 参数说明：
    # - n_e: 嵌入向量的数量（离散码本的大小），类型：int
    # - e_dim: 每个嵌入向量的维度，类型：int
    # - beta: 用于计算损失时的平衡系数，类型：float
    # - z_channels: 输入通道数，用于1x1卷积调整通道数，类型：int
    # - remap: 可选，指定索引重映射文件的路径，类型：str或None
    # - unknown_index: 当 remap 存在但找不到对应索引时如何处理，"random"或"extra"或整数
    # - sane_index_shape: 是否保证输出的索引形状较为合理，类型：bool
    # - legacy: 是否使用旧版本BUG逻辑，类型：bool
    # - use_voxel: 是否处理体素数据，类型：bool（本示例中固定为True）
    # -------------------------------------------------------------------
    def __init__(self, n_e, e_dim, beta, z_channels, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, use_voxel=True):
        super().__init__()  # 调用父类nn.Module的初始化函数
        self.n_e = n_e  # 记录码本的数量（嵌入向量数）
        self.e_dim = e_dim  # 记录每个嵌入向量的维数
        self.beta = beta  # 损失中平衡重构误差的参数
        self.legacy = legacy  # 记录是否使用legacy版本

        # 初始化一个嵌入层，把每个索引映射到一个e_dim维的向量上
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 对嵌入权重进行均匀初始化，范围为[-1/n_e, 1/n_e]
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # 如果传递了remap参数，则加载外部指定的映射文件
        self.remap = remap
        if self.remap is not None:
            # 使用np.load加载 remap 文件，然后转换为 torch.tensor，并注册为buffer（不作为模型参数）
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # 重新计算重映射后的嵌入数量
            self.re_embed = self.used.shape[0]  # used 的shape一般是 [N]，N 表示重映射后的条目数
            self.unknown_index = unknown_index  # unknown_index指明了若未知索引如何处理
            if self.unknown_index == "extra":
                # 如果设置的是"extra"，则把未知索引设置成一个额外的索引，即放在re_embed末尾
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1  # 新增一个额外的索引位置
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            # 没有remap时，直接保持嵌入个数
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape  # 记录是否调整索引输出的形状

        # 选择卷积类，如果use_voxel为True，就用Conv3d（3D卷积）处理体素数据，否则使用Conv2d
        # 本示例硬编码为True，所以使用Conv3d
        conv_class = torch.nn.Conv3d if True else torch.nn.Conv2d
        # 用1x1卷积层将输入z转变为与嵌入向量维度一致的特征图（通过改变通道数来匹配e_dim）
        self.quant_conv = conv_class(z_channels, self.e_dim, 1)
        # 量化后再通过1x1卷积层将嵌入向量转换回输入通道数
        self.post_quant_conv = conv_class(self.e_dim, z_channels, 1)

    # -------------------------------------------------------------------
    # 函数：remap_to_used
    # 目的：将原始的最小编码索引映射到 remap 指定的索引空间中
    # 说明：输入inds是一个张量（通常是int类型，形状至少两维[batch, ...]），
    #      将其重排列为二维，再匹配 remap 中记录的可用索引，用于后处理索引映射。
    # -------------------------------------------------------------------
    def remap_to_used(self, inds):
        ishape = inds.shape  # 记录输入的形状，通常至少为(b, spatial...)
        assert len(ishape) > 1, "输入维度至少需要两维（batch和其他空间维）"  # 必须多于1维
        inds = inds.reshape(ishape[0], -1)  # 将空间维度展平，使得inds形状变为 (batch, num_elements)
        used = self.used.to(inds)  # 将 buffer used 数据移到与inds相同的设备、类型
        # match: (batch, num_elements, used_len) 布尔匹配矩阵，判断每个索引是否存在于used中
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # 对第三个维度求argmax，得到匹配的位置，即重映射后的索引
        new = match.argmax(-1)
        # unknown: 判断哪些位置未匹配到任何used中的条目；sum(2)<1表示整行都为0
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            # 对未知位置随机采样一个索引，范围在[0, re_embed)；输出tensor与new中相同设备
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            # 否则直接将未知位置设置为给定的unknown_index（此时应为整数）
            new[unknown] = self.unknown_index
        return new.reshape(ishape)  # 恢复到输入的原始形状

    # -------------------------------------------------------------------
    # 函数：unmap_to_all
    # 目的：将 remap 后的索引还原到初始的完整码本索引（所有embedding），用于反向恢复
    # -------------------------------------------------------------------
    def unmap_to_all(self, inds):
        ishape = inds.shape  # 记录输入索引shape，一般为(b, spatial...)
        assert len(ishape) > 1, "输入维度至少需要两维"
        inds = inds.reshape(ishape[0], -1)  # 平展除batch以外的其它维度
        used = self.used.to(inds)  # 将 used 转换到与 inds 相同设备上
        if self.re_embed > self.used.shape[0]:  # 如果re_embed中包含extra token（额外指定的未知索引）
            inds[inds >= self.used.shape[0]] = 0  # 将超过used索引最大范围的索引设为0
        # 使用 gather 操作恢复原始索引。其中 used[None, :] 扩展第一个维度重复batch次
        back = torch.gather(used[None, :].expand(inds.shape[0], -1), 1, inds)
        return back.reshape(ishape)  # 恢复原始shape

    # -------------------------------------------------------------------
    # 函数：forward
    # 目的：实现向量量化模块的前向传播
    # 说明：先通过一层卷积将输入特征映射到嵌入向量的维度（e_dim），
    #      然后通过 forward_quantizer 进行量化，将连续值转为离散码本表示，
    #      最后再通过另一层卷积映射回原始通道数（z_channels）。

    # 关于参数：
    # - z: 输入张量，其shape可能为 [batch, z_channels, depth, height, width]，例如 torch.Size([1, 4, 30, 50, 1])
    # - temp, rescale_logits, return_logits, is_voxel：提供接口参数，但目前只用来适配Gumbel等接口（这里固定部分参数值）
    # -------------------------------------------------------------------
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        # z的shape举例： torch.Size([1, 4, 30, 50, 1])
        z = self.quant_conv(z)  # 应用1x1卷积：输入z的shape从 [b, z_channels, d, h, w] 转换为 [b, e_dim, d, h, w]
        # 调用 forward_quantizer 对映射到 e_dim 的特征进行向量量化，返回量化后的输出、损失、其他信息
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.forward_quantizer(z, temp, rescale_logits, return_logits, is_voxel)
        z_q = self.post_quant_conv(z_q)  # 通过1x1卷积将量化输出映射回 z_channels，恢复到原始通道数
        # 返回量化后的重建结果 z_q、量化损失loss以及其他量化相关信息的元组
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # -------------------------------------------------------------------
    # 函数：forward_quantizer
    # 目的：实现真正的向量量化逻辑，对于输入特征进行编码（找到最近的码本向量），
    #      并计算量化损失；核心步骤包括：
    #        1. 重排列输入张量，将通道维调整到最后方便计算嵌入距离
    #        2. 将输入展平后计算与所有码本向量的欧氏距离（通过平方距离公式）
    #        3. 选择距离最小的码本向量作为编码结果，计算量化损失
    #        4. 保存梯度信息并恢复原输入的形状
    #
    # 关于变量数据类型和形状：
    # - z: 输入经过 quant_conv 后 shape 为 [b, e_dim, d, h, w]
    # - z经过 rearrange 后 shape 变为 [b, d, h, w, e_dim] （转换到最后一维为embedding维度）
    # - z_flattened: 2D张量，shape为 [b * d * h * w, e_dim]
    # - self.embedding.weight: 嵌入矩阵，shape为 [n_e, e_dim]
    # - d: 距离矩阵，shape为 [b * d * h * w, n_e]
    # -------------------------------------------------------------------
    def forward_quantizer(self, z, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        # 以下断言保证传入的接口参数符合预期，目前仅支持temp==1.0, rescale_logits=False, return_logits=False
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # 将输入 z 重排列，从 [b, e_dim, d, h, w] 转换为 [b, d, h, w, e_dim]
        # 具体使用einops.rearrange： 'b c d h w -> b d h w c'（若处理2D图像则另有一套排列）
        if not True:  # 若处理2D情况
            z = rearrange(z, 'b c h w -> b h w c').contiguous()  # 保证内存连续
        else:  # 当前处理3D（体素）情况
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        # 将 z 展平成二维张量，形状 [batch_size * d * h * w, e_dim]
        z_flattened = z.view(-1, self.e_dim)
        # 计算每个向量 z 与所有嵌入向量 e_j 的欧氏距离：
        # 考虑公式 (z - e)^2 = z^2 + e^2 - 2 * z * e
        # 1. torch.sum(z_flattened ** 2, dim=1, keepdim=True): 计算 z^2，对每行求和，shape [N, 1]
        # 2. torch.sum(self.embedding.weight**2, dim=1): 对每个码本向量计算平方和，shape [n_e]
        # 3. torch.einsum('bd,dn->bn', ...): 计算z与e的点积，结果形状为 [N, n_e]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        # 根据距离d，选取每个z最接近的码本向量，其索引作为量化编码；min_encoding_indices的shape为 [N]
        min_encoding_indices = torch.argmin(d, dim=1)
        # 用最小编码索引，在嵌入层中查找对应的嵌入向量；随后将结果恢复成 z 的形状
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None  # 此处可以计算量化器熵的困惑度指标，这里暂不计算
        min_encodings = None  # 用于返回最小编码的分布（有时用于监控），此处未使用

        # 计算嵌入损失（Vector Quantization Loss）
        # 注：legacy表示旧版本BUG逻辑的顺序不同
        if not self.legacy:
            # 这里对比的保真损失包括两部分，beta项的权重放在z_q.detach()-z上
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            # legacy版本中，将beta项加在后半部分（注意：detach()用于不传递梯度）
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # 保留梯度信息，写法为“z + (z_q - z).detach()”相当于向前传播时使用 z_q，
        # 但梯度流却绕过z_q，直接传递给 z，从而解决离散化梯度无法传播的问题（直通操作技巧）
        z_q = z + (z_q - z).detach()

        # 将量化后的张量z_q重排列回原始的网络输出形状：
        # 如果是2D图像，则排列为 [b, e_dim, h, w]；否则排列为 [b, c, d, h, w]
        if not True:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        else:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()
        
        # 如果使用 remap，则对最小编码索引进行 remap 处理
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # 展平除batch维以外的其它维度
            min_encoding_indices = self.remap_to_used(min_encoding_indices)  # 调用 remap 函数
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # 再次展平成二维，形状为 [N,1]

        # 如果设置了 sane_index_shape，则根据是否处理体素数据对索引形状进行调整
        if self.sane_index_shape:
            if not is_voxel:  # 如果不是体素（2D图像情况）
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3])
            else:  # 体素情况，形状应为 [batch, d, h, w]
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])
                
        # 返回量化后的张量 z_q, 损失值 loss, 以及其他信息组成的元组（这里 perplexity, min_encodings 未计算）
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # -------------------------------------------------------------------
    # 函数：get_codebook_entry
    # 目的：根据给定的索引和形状从码本中获取对应的嵌入向量
    # 参数：
    # - indices：量化后的索引，类型通常为 LongTensor
    # - shape：指定输出张量的形状，通常为 (batch, height, width, channel)
    # -------------------------------------------------------------------
    def get_codebook_entry(self, indices, shape):
        # 若使用 remap，则先恢复到完整的codebook索引
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # 加入batch轴，展平空间维度
            indices = self.unmap_to_all(indices)  # 调用unmap函数返回原始index
            indices = indices.reshape(-1)  # 将结果重新展平成一维
        # 通过嵌入表查找相应的量化向量，返回的z_q的shape为 [num_indices, e_dim]
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)  # 将z_q reshape为指定shape（例如 [batch, height, width, channel]）
            # 对 z_q 进行维度permute，将通道维转到第二个维度，符合常见网络输入格式 [batch, channel, height, width]
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q  # 返回码本中的嵌入向量

    # -------------------------------------------------------------------
    # 函数：get_codebook_index
    # 目的：对于输入的z（经过 Conv 变换后的特征图）计算与码本中最接近的索引，
    #      基本过程与 forward_quantizer 一致，但只返回索引信息（不返回重构后的向量和损失）
    # 参数：
    # - z: 输入张量，形状为 [b, z_channels, d, h, w]
    # - is_voxels: 是否处理体素数据，默认为False，此处硬编码为True
    # -------------------------------------------------------------------
    def get_codebook_index(self, z, is_voxels=False):
        # 首先对输入z使用 quant_conv 将通道数从原始 z_channels 转换为 e_dim
        z = self.quant_conv(z)
        if not True:  # 若处理2D图像
            b, c, h, w = z.shape
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        else:  # 处理3D体素数据
            b, c, d, h, w = z.shape
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        # 将重排列后的 z 展平成二维，形状为 [b * d * h * w, e_dim]
        z_flattened = z.view(-1, self.e_dim)
        # 计算z_flattened与码本所有向量的欧氏距离，与 forward_quantizer 中方法一致
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        # 选择每个输入向量对应的最小距离码本的索引，shape为 [b*d*h*w]
        min_encoding_indices = torch.argmin(d, dim=1)
        if not True:  # 2D情况 reshape 为 [batch, height, width]
            min_encoding_indices = min_encoding_indices.reshape(b, h, w)
        else:
            # 3D体素情况，reshape为 [batch, d, h, w]；注意这里d来自前面已提取的d变量
            min_encoding_indices = min_encoding_indices.reshape(b, d, h, w)
        return min_encoding_indices  # 返回每个位置对应的码本索引