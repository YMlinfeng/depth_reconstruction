import torch
from thop import profile
import types

# 导入你的模型定义
from vqvae.vae_2d_resnet import VAERes2DImgDirectBC

# 构造一个模拟的 args 对象
# 如果你的 args 中除了下面几个参数外还有其他需要用到的属性，
# 请相应地增加进去，或者从实际使用中复制一个 args 对象。
args = types.SimpleNamespace(
    inp_channels=80,
    out_channels=80,
    mid_channels=1024,
    z_channels=4,
    # 如果模型的 forward 里需要其他属性也加入，比如：
    input_height=60,
    input_width=100,
    # 其他可能需要的属性……
)

# 实例化模型
vqvae = VAERes2DImgDirectBC(
    inp_channels=args.inp_channels,
    out_channels=args.out_channels,
    mid_channels=args.mid_channels,
    z_channels=args.z_channels,
    vqvae_cfg=None  # 若有额外配置可在此传入
)

# 将模型放到合适的设备上（例如 CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae.to(device)

# 构造 dummy 输入数据
# 注意：这里的输入尺寸需要与你模型 forward 函数中期望的尺寸一致。
# 根据你的代码训练时传入的 voxel 的形状注释来看，这里假设输入 tensor 的形状为 [B, inp_channels, D, H, W]，
# 你可以根据实际情况对 D, H, W 做相应调整。
dummy_input = torch.randn(1, args.inp_channels, 60, 100, 1, device=device)

# 调用 thop 的 profile 进行统计
# 注意，由于你的模型的 forward 定义为接收两个参数（dummy_input 和 args），
# 因此在 inputs 参数中传入一个元组 (dummy_input, args)
macs, params = profile(vqvae, inputs=(dummy_input, args), verbose=False)

print("模型参数总量: {:,}".format(params))
print("MACs: {:,}".format(macs))