import torch  # 导入 PyTorch 库，用于张量操作和神经网络构建（Tensor类型：torch.Tensor）
import torch.nn.functional as F  # 导入常用的神经网络函数，如激活函数、损失函数（例如 F.relu, F.mse_loss 等）
import importlib  # 导入 importlib 模块，用于动态导入模块
from einops import rearrange  # 导入 rearrange 函数，可灵活改变张量的形状，例如将 (B, C, H, W) 变为 (B, H, W, C)
from torch.nn import Embedding  # 导入 Embedding 层，用于构造词嵌入表（例如码本）
from vqganlc.models.discriminator import NLayerDiscriminator, weights_init  # 导入鉴别器模型和权重初始化函数
from vqganlc.models.lpips import LPIPS  # 导入 LPIPS 模块，用于感知相似度计算（衡量高层特征差异）
from vqganlc.models.encoder_decoder import Encoder, Decoder, Decoder_Cross  # 导入编码器/解码器模块

# ----------------------------------------------------------------------------
# 模块说明：通过字符串获取对象实例（从动态加载的模块中返回类/函数）
# 主要用于根据配置文件中的字符串路径动态实例化模块
# ----------------------------------------------------------------------------
def get_obj_from_str(string, reload=False):
    # 参数 string: 字符串形式的模块路径和类名，例如 "module.submodule.ClassName"
    module, cls = string.rsplit(".", 1)  # 将字符串按最后一个点分隔，module（模块名），cls（类名）
    if reload:
        module_imp = importlib.import_module(module)  # 导入模块
        importlib.reload(module_imp)  # 重新加载模块，确保得到最新状态
    # 返回模块中对应的属性（一般是类），用于后续用关键字参数实例化对象
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    """
    宏观目的：
    根据传入的配置字典 config【包含 'target' 和 可选 'params'】实例化对应的对象。
    config 示例：{'target': 'module.submodule.ClassName', 'params': {'param1': value1, ...}}
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")  # 键不存在则报错
    # 使用 get_obj_from_str 获得目标类，并展开参数字典(**params)进行实例化。
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# ----------------------------------------------------------------------------
# 模块说明：定义 VQGAN-LC 模型类
#
# 该模型主要包含：
# 1. 编码器（Encoder）：接受输入图像并转换成潜在空间的表示
# 2. 量化器（Quantizer）：将连续的潜在表示映射到离散的码本嵌入上
# 3. 解码器（Decoder）：将离散码转换回图像
# 4. 鉴别器（Discriminator）：用于计算对抗损失，提高图像生成质量
#
# 同时包含了 EMA（指数移动平均）机制的实现和自适应权重计算，用于平衡
# 重构损失、量化损失、感知损失以及对抗损失。
# ----------------------------------------------------------------------------
class VQModel(torch.nn.Module):
    def __init__(self,
                 args,  # 参数对象，包含模型训练和结构的配置属性（例如 stage、embed_dim、n_vision_words、quantizer_type 等）
                 ddconfig,  # 字典格式的配置，指定 Encoder/Decoder 的网络结构（例如 z_channels、分辨率、层数等）
                 embed_dim,  # 嵌入维度（整数），例如256、512，决定了潜在空间和码本的维度
                 ckpt_path=None,  # 可选的预训练模型文件路径，用于加载已有的权重
                 ignore_keys=[],  # 加载权重时需要忽略的键列表
                 image_key="image",  # 输入数据字典中图像对应的键名
                 colorize_nlabels=None,  # 如果需要颜色映射，该参数指定标签的数量（整数）
                 monitor=None,  # 用于模型监控或者调试的对象（可选）
                 remap=None,  # 可选参数，若需要重新映射输出编码索引则会用到
                 sane_index_shape=False,  # 标志，若 True，则量化器返回的索引形状为 (batch, height, width)
                 ):
        super().__init__()
        self.image_key = image_key  # 保存输入数据的图像键
        self.args = args  # 保存参数对象
        
        self.stage = args.stage  # 训练阶段：例如 1 表示重构训练，2 表示 GPT training 阶段
        self.encoder = Encoder(**ddconfig)  # 使用 ddconfig 初始化编码器；输入图像 shape: (B, 3, H, W)
        self.decoder = Decoder(**ddconfig)  # 使用 ddconfig 初始化解码器；输出图像 shape 应与输入匹配
        # 初始化鉴别器，输入为 3 通道图像，采用两层卷积（n_layers=2），不使用 actnorm
        self.discriminator = NLayerDiscriminator(input_nc=3,
                                                 n_layers=2,
                                                 use_actnorm=False,
                                                 ndf=64
                                                ).apply(weights_init)  # 对鉴别器参数进行权重初始化
        
        embed_dim = args.embed_dim  # 从参数中重新获得嵌入向量维度
        self.perceptual_loss = LPIPS().eval()  # 初始化 LPIPS 模块，用于计算感知损失，设置为 eval() 模式避免训练
        self.perceptual_weight = args.rate_p  # 感知损失的权重因子，用于 loss 加权
        self.quantize_type = args.quantizer_type  # 量化器类型，例如 "ema"（指数移动平均）等

        print("****Using Quantizer: %s" % (args.quantizer_type))
        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失，一般用于分类任务，此处可能用于码本预测
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)  # 如果有预训练权重，则加载
        self.image_key = image_key
        if colorize_nlabels is not None:
            # 若提供 colorize_nlabels，则生成一个随机的颜色映射矩阵，形状为 (3, colorize_nlabels, 1, 1)
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor  # 保存监控变量
        
        # 初始化码本（词嵌入），codebook_dim 即码本向量的维度，初始与 embed_dim 相同
        codebook_dim = embed_dim
        if args.tuning_codebook == -1:  # Random：随机初始化且允许调优
            print("****Using Tuned Random Codebook****")
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)  # 构造一个大小为 (n_vision_words, embed_dim) 的嵌入表
            # 均匀初始化，数值范围为 (-1/n_vision_words, 1/n_vision_words)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = True  # 允许梯度更新
            
        elif args.tuning_codebook == -2:  # Random Fix：随机初始化但固定，不调优
            print("****Using Fix Random Codebook****")
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = False  # 固定，不更新梯度

        elif args.tuning_codebook == 0:  # Fix Initialized Codebook：加载预训练权重并固定
            print("****Using Fix Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]  # 更新词数量为 checkpoint 的行数
            codebook_dim = checkpoint.shape[1]  # 更新码本向量维度
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint  # 加载预训练 embedding
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = False  # 固定权重

        elif args.tuning_codebook == 1:  # Tuning Initialized Codebook：加载预训练权重但允许调优
            print("****Tuning Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]
            codebook_dim = checkpoint.shape[1]
            print("Word Number:%d" % (args.n_vision_words))
            print("Feature Dim:%d" % (embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = True  # 允许调优

        self.e_dim = embed_dim  # 保存嵌入维度，后续用于重构 z 张量
        self.remap = remap  # 保存 remap 信息（若需要对量化后的索引进行修改）
        self.sane_index_shape = sane_index_shape  # 标志是否返回 (B, H, W) 形状的索引
        
        # 定义量化前的卷积层：将编码器输出的通道数转换到嵌入维度 embed_dim
        # 输入: (B, z_channels, H, W) ； 输出: (B, embed_dim, H, W)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # 定义反量化卷积层，将嵌入空间映射回编码器原始的 z_channels
        # 输入: (B, embed_dim, H, W) ； 输出: (B, z_channels, H, W)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # 如果需要进一步将码本映射到嵌入空间，可以选择线性（Linear）或 MLP 投影器
        if args.use_cblinear == 1:
            print("****Using Linear Codebook Projector****")
            self.codebook_projection = torch.nn.Linear(codebook_dim, embed_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=embed_dim ** -0.5)
        elif args.use_cblinear == 2:
            print("****Using MLP Codebook Projector****")
            self.codebook_projection = torch.nn.Sequential(
                torch.nn.Linear(codebook_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, embed_dim),
            )
            # 注意：此处MLP的权重初始化可根据需要调整

        # 若量化器类型选择 "ema"，则需要额外的变量保存指数移动平均信息
        if self.quantize_type == "ema":
            self.decay = 0.99  # EMA 衰减因子
            self.eps = 1e-5    # 防止除零的数值稳定参数
            # cluster_size: 记录每个码本向量在当前 mini-batch 中被选中的次数，shape: (n_vision_words,)
            self.cluster_size = torch.nn.Parameter(torch.zeros(args.n_vision_words), requires_grad=False)
            # embed_avg: 每个码本向量的累积和，用于 EMA 更新，形状: (n_vision_words, embedding_dim)
            self.embed_avg = torch.nn.Parameter(self.tok_embeddings.weight.clone(), requires_grad=False)
            self.update = True  # 标记 EMA 更新是否进行
            self.tok_embeddings.weight.requires_grad = False  # EMA 方式下不通过反向传播更新码本
            self.num_tokens = args.n_vision_words  # 保存码本中总的词数

    # ----------------------------------------------------------------------------
    # 下方的一系列函数实现了模型的不同组成部分：
    # 1. hinge_d_loss：用于计算鉴别器的 hinge 损失
    # 2. calculate_adaptive_weight：通过比较梯度范数计算自适应损失权重
    # 3. cluster_size_ema_update / embed_avg_ema_update / weight_update：实现 EMA 更新码本的统计值
    # 4. quantize：核心量化函数，将连续潜在向量映射到离散码本，并计算量化损失
    # 5. forward：前向传播，依据训练阶段（生成器或鉴别器更新）计算不同的损失
    # 6. encode / decode：分别实现编码器和解码器的操作
    # 7. get_last_layer / decode_code：辅助函数，用于获取解码器末层和根据码本生成图像
    # ----------------------------------------------------------------------------

    def hinge_d_loss(self, logits_real, logits_fake):
        # 宏观目的：计算鉴别器损失，鼓励真实图像的判别值大于1，生成图像小于-1
        # logits_real: Tensor，形状通常为 (B, *)，代表真实图像的输出
        # logits_fake: Tensor，生成图像的输出
        loss_real = torch.mean(F.relu(1. - logits_real))  # 对真实图像的 hinge 损失，期望 logits_real >= 1
        loss_fake = torch.mean(F.relu(1. + logits_fake))  # 对生成图像的 hinge 损失，期望 logits_fake <= -1
        d_loss = 0.5 * (loss_real + loss_fake)  # 平均两个损失作为最终鉴别器损失
        return d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, discriminator_weight, last_layer=None):
        # 宏观目的：平衡重构损失 (nll_loss) 与生成器损失 (g_loss) 的权重
        # 通过比较 last_layer 上的梯度范数来自适应调整鉴别器损失的贡献，确保各项损失处于同一数量级
        # 其中:
        #   nll_loss: 重构相关损失（例如 L1 或 L2 损失）
        #   g_loss: 生成器的对抗损失（由鉴别器反馈）
        #   discriminator_weight: 用户设定的权重因子
        #   last_layer: 通常为解码器最后一层（例如 conv_out）的权重 tensor，形状与实际层权重一致
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]  # 求 nll_loss 对 last_layer 的梯度
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]  # 求 g_loss 的梯度

        # 计算梯度的 L2 范数（即各自张量元素的平方和开根号）
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()  # 限制 d_weight 范围，同时 detach 避免反向传播
        d_weight = d_weight * discriminator_weight  # 乘上鉴别器的权重因子
        return d_weight

    def cluster_size_ema_update(self, new_cluster_size):
        # 宏观目的：用 EMA 更新码本索引的累积统计，帮助稳定码本更新
        # new_cluster_size: 新的簇大小统计，Tensor 形状：(n_vision_words,)
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        # 宏观目的：更新码本向量对应的累积和，后续将归一化更新权重
        # new_embed_avg: 新的嵌入向量求和，Tensor 形状：(n_vision_words, embedding_dim)
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        # 宏观目的：根据 EMA 更新后的统计结果归一化码本得嵌入向量，然后复制到 tok_embeddings 中
        n = self.cluster_size.sum()  # 汇总所有码本向量累计使用次数，标量
        # smoothed_cluster_size: 将 cluster_size 归一化并平滑，形状：(n_vision_words,)
        smoothed_cluster_size = ((self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n)
        # 对 embed_avg 进行归一化，每个码本向量除以对应的 smoothed_cluster_size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)  # unsqueeze 将 (n,) 转为 (n,1)
        self.tok_embeddings.weight.data.copy_(embed_normalized)  # 更新嵌入权重

    def quantize(self, z, temp=None, rescale_logits=False, return_logits=False):
        """
        宏观目的：
        将编码器输出的连续张量 z 量化为离散码本向量，并计算量化损失。
        
        参数：
          z: Tensor，编码器输出经过 quant_conv 后的特征，形状为 (B, C, H, W)，其中 C==embed_dim
          temp, rescale_logits, return_logits: 保留参数（当前未详细使用）
          
        主要步骤：
          1. 重新排列 z 的形状为 (B, H, W, C)，方便后续计算
          2. 将 z 展平为二维张量 (B*H*W, C)
          3. 计算每个 z 向量与所有码本向量之间的欧几里得距离平方
          4. 根据不同量化策略（例如 ema 或非 ema）进行对应处理
          5. 使用直通估计器（gradient trick）保留反向传播梯度
          6. 将量化后的张量恢复到原输入形状，并（若需要）对索引进行 remap
        """
        # 调整 z 的形状：原始 (B, C, H, W) => (B, H, W, C)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        # 将 (B, H, W, C) 重塑为 (B*H*W, C) 方便与码本向量比较
        z_flattened = z.view(-1, self.e_dim)

        # 如果使用代码本投影，则将 tok_embeddings.weight 经过投影层映射到 embed_dim
        if self.args.use_cblinear != 0:
            tok_embeddings_weight = self.codebook_projection(self.tok_embeddings.weight)
        else:
            tok_embeddings_weight = self.tok_embeddings.weight

        # 计算每个 z 向量与所有码本向量之间的欧氏距离的平方
        # 公式：||z - e||^2 = ||z||^2 + ||e||^2 - 2 * (z · e)
        # z_flattened: (B*H*W, C)； tok_embeddings_weight: (n_tokens, C)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(tok_embeddings_weight ** 2, dim=1) - \
            2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(tok_embeddings_weight, 'n d -> d n'))
        # 得到距离矩阵 d，形状: (B*H*W, n_tokens)

        # 对每个 z 向量选取距离最小的码本索引
        min_encoding_indices = torch.argmin(d, dim=1)  # 形状: (B*H*W,)

        # 针对不同的量化策略：EMA 或 非 EMA
        if self.quantize_type == "ema":
            # 获取对应的量化结果 z_q: 通过 embedding 取出向量，重塑为 (B, H, W, C)
            z_q = self.tok_embeddings(min_encoding_indices).view(z.shape)
            # 构造 one-hot 编码矩阵，形状: (B*H*W, n_tokens)
            encodings = F.one_hot(min_encoding_indices, self.num_tokens).type(z.dtype)
            # 求各个码本被选中的平均概率，shape: (n_tokens,)
            avg_probs = torch.mean(encodings, dim=0)
            # 计算困惑度（perplexity），衡量码本的使用多样性
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))
            min_encodings = None
            # EMA 更新：累计每个码本向量的选取次数（encodings_sum shape：(n_tokens,)）
            encodings_sum = encodings.sum(0)
            self.cluster_size_ema_update(encodings_sum)
            # 计算当前 mini-batch 中，各码本向量对应的嵌入向量总和
            embed_sum = encodings.transpose(0,1) @ z_flattened  # shape: (n_tokens, embedding_dim)
            self.embed_avg_ema_update(embed_sum)
            # 根据 EMA 更新后的统计归一化 tok_embeddings 权重
            self.weight_update(self.num_tokens)
            # 计算量化损失，使用 MSE 损失，即 z_q 与 z 之间的均方误差（detach 保证非梯度传递）
            loss = F.mse_loss(z_q.detach(), z)
        else:
            # 非 EMA 情况下，直接根据最小编码索引取得嵌入向量
            min_encodings = None
            perplexity = None
            # 利用 F.embedding 获取结果，重塑回 (B, H, W, C)
            z_q = F.embedding(min_encoding_indices, tok_embeddings_weight).view(z.shape)
            # 量化损失：结合两部分损失，部分项使用 detach 保证梯度独立传播
            loss = torch.mean((z_q.detach() - z)**2) + 0.33 * torch.mean((z_q - z.detach()) ** 2)

        # 直通估计器技巧：使 z_q 在梯度反向传播时和 z 保持一致（实际上梯度来自 z）
        z_q = z + (z_q - z).detach()

        # 恢复张量形状为 (B, C, H, W)，与输入一致
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        # 如果定义了 remap，则对编码索引进行重塑及映射
        if self.remap is not None:
            # 先将 min_encoding_indices reshape 为 (B, -1)
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)
        
        # 如果 sane_index_shape 为 True，则将索引 reshape 为 (B, H, W)
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        # 返回：量化后的张量 z_q，量化损失 loss，以及额外信息 (距离矩阵 d, min_encodings, 最终编码索引)
        return z_q, loss, (d, min_encodings, min_encoding_indices)
    
    def forward(self, input, global_input, data_iter_step, step=0, is_val=False):
        """
        宏观目的：
        前向传播函数，根据不同训练阶段和步骤计算以下内容：
          - 进行完整的 VQGAN 计算：编码->量化->解码，计算重构损失、感知损失、以及对抗损失
        参数：
          input: 输入图像，Tensor，形状 (B, C, H, W, D) = (B, 4, 60, 100, 20)
          global_input: 全局条件输入（比如用于条件生成，具体用途依任务而定）
          data_iter_step: 当前迭代步数，用于调度鉴别器更新时机
          step: 训练步骤标志（0 为生成器更新，非 0 时为鉴别器更新）
          is_val: 是否为验证模式，控制是否计算对抗损失
        """
        # 先对输入图像进行编码和量化，返回：
        #  quant: 量化后的潜在表示，形状 (B, embed_dim, H', W')
        #  qloss: 量化损失（标量）
        #  tk_labels: 量化后对应的码本索引，原始形状为 (B*H'*W',) 或 (B, H', W') 依据 sane_index_shape
        quant, qloss, [_, _, tk_labels] = self.encode(input)
        
        # 对量化后的特征进行解码，生成重构图像 dec，shape 应为 (B, 3, H, W)
        dec = self.decode(quant)

        # 计算重构损失：均值 L1 损失（输入与重构图像的绝对误差的均值）
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        
        # 计算感知损失：LPIPS，用于衡量两幅图像在深度特征空间的差异
        p_loss = torch.mean(self.perceptual_loss(input.contiguous(), dec.contiguous()))
        
        if step == 0:  # 生成器更新阶段
            # 通过鉴别器对生成的图像 dec 进行预测，获取 logits，通常形状为 (B, 1) 或 (B, N)
            logits_fake = self.discriminator(dec)
            g_loss = -torch.mean(logits_fake)  # 生成器希望提高鉴别器预测值，所以取负值

            if is_val:
                # 验证时不进行对抗损失更新，只返回各项损失的加权和
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + 0 * g_loss
                return loss, rec_loss, qloss, p_loss, g_loss, tk_labels.view(input.shape[0], -1), dec
            
            # 计算自适应权重，根据解码器最后一层的梯度比较生成器损失与重构损失的比例
            d_weight = self.calculate_adaptive_weight(
                rec_loss + self.perceptual_weight * p_loss,
                g_loss,
                self.args.rate_d,
                last_layer=self.decoder.conv_out.weight
            )
            
            # 根据迭代步数决定是否启用对抗损失项（例如 disc_start 之后）
            if data_iter_step > self.args.disc_start:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + d_weight * g_loss
            else:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + 0 * g_loss

            return loss, rec_loss, qloss, p_loss, g_loss, tk_labels, dec
        else:  # 鉴别器更新阶段
            # 对输入真实图像和生成图像均使用 detach() 防止梯度回传到生成器
            logits_real = self.discriminator(input.contiguous().detach().clone())
            logits_fake = self.discriminator(dec.detach().clone())
            # 计算 hinge 损失，鼓励真实样本输出较高值，生成样本输出较低值
            d_loss = self.hinge_d_loss(logits_real, logits_fake)
            loss = d_loss + 0 * (rec_loss + qloss + p_loss)  # 鉴别器更新中只关心 d_loss

            return loss, rec_loss, qloss, p_loss, d_loss, tk_labels, dec

    def encode(self, input):
        """
        宏观目的：
        1. 使用编码器将图像输入转换为中间特征表示
        2. 用量化卷积层 (quant_conv) 将特征转换到码本嵌入维度上
        3. 调用量化函数将连续特征映射到离散码本上，输出量化后张量和量化损失
          
        参数：
          input: 输入图像 Tensor，形状 (B, 3, H, W)
        返回：
          quant: 量化后的潜在向量，形状 (B, embed_dim, H', W')
          emb_loss: 量化损失，标量 Tensor
          info: 其它信息（例如距离矩阵、编码索引等）
        """
        h = self.encoder(input)  # 编码器输出，形状依 ddconfig 决定，通常为 (B, z_channels, H', W')
        h = self.quant_conv(h)  # 通过量化卷积映射到嵌入空间，输出形状 (B, embed_dim, H', W')
        if self.e_dim == 768 and self.args.tuning_codebook != -1:
            # 若 embed_dim 为768，则对特征进行 L2 归一化，归一化后每个特征向量的 L2 范数为 1
            h = h / h.norm(dim=1, keepdim=True)
        quant, emb_loss, info = self.quantize(h)  # 调用量化函数
        return quant, emb_loss, info

    def decode(self, quant, global_c_features=None):
        """
        宏观目的：
        将量化后的潜在向量映射回原始图像空间
        步骤：
          1. 先通过 post_quant_conv 将嵌入向量映射到编码器对应的通道数
          2. 再通过解码器还原成图像
          
        参数：
          quant: 量化后的张量，形状 (B, embed_dim, H', W')
        返回：
          dec: 解码还原后的图像，形状 (B, 3, H, W)
        """
        quant = self.post_quant_conv(quant)  # 映射回 z_channels，形状: (B, z_channels, H', W')
        dec = self.decoder(quant)  # 解码器还原图像
        return dec
    
    def get_last_layer(self):
        # 返回解码器最后一层卷积层的权重，用于自适应权重计算
        return self.decoder.conv_out.weight

    def decode_code(self, code_b):
        # 根据给定的码本索引张量 code_b，利用量化层中的 embedding 转换为嵌入向量，再解码为图像
        quant_b = self.quantize.embedding(code_b)
        dec = self.decode(quant_b)
        return dec