#!/usr/bin/env bash
##############################################################################
# 说明：
#  1) 需在每个节点上执行同样的命令，但传入不同的 node_rank（从0到 num_nodes-1）。
#  2) 如果集群有 2～8 个节点，每节点 8 张 GPU，则总 GPU 数 = num_nodes * 8。
#  3) MASTER_ADDR、MASTER_PORT 要设置成主节点（node_rank=0 这台机器）的 IP / 主机名 + 端口。
#  4) 您的代码文件仍然是 my_train_vqvae.py（或 main.py），无需额外改动。
#  5) 如果您有更复杂的启动逻辑（如 Slurm 脚本），可在此脚本中合并 srun 等命令。
##############################################################################

### 配置部分 ###
NUM_NODES=1           # 节点总数（可改为 2～8）
GPUS_PER_NODE=8       # 每台节点上有 8 张 GPU
NODE_RANK=0           # 当前节点的 rank：0 ~ (NUM_NODES-1)
MASTER_ADDR="10.124.2.134"   # 主节点的地址或主机名（请务必改成集群实际的主机名/IP）
MASTER_PORT=12355     # 主节点进程通信使用的端口（任意一个空闲端口即可）

# =============== 可按需在此传入更多参数 ==============
TRAIN_FILE="train2.py"  
TRAIN_ARGS="--batch_size 4 --epochs 20 --lr 1e-4"  # 需要传入的其他命令行参数
# ======================================================

### 打印当前配置信息 ###
echo "Start training with PyTorch Distributed:"
echo "  NUM_NODES     = $NUM_NODES"
echo "  GPUS_PER_NODE = $GPUS_PER_NODE"
echo "  NODE_RANK     = $NODE_RANK"
echo "  MASTER_ADDR   = $MASTER_ADDR"
echo "  MASTER_PORT   = $MASTER_PORT"
echo "  TRAIN_FILE    = $TRAIN_FILE"
echo "  TRAIN_ARGS    = $TRAIN_ARGS"
echo "===================================================="

### 启动分布式训练 ###
# torchrun 是 PyTorch 2.0 官方推荐的分布式启动器
# 如果是 PyTorch<1.9，可能需要使用 python -m torch.distributed.launch

torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $TRAIN_FILE $TRAIN_ARGS

##############################################################################
# 用法示例：
#  1) 在主节点 (node_rank=0) 上：
#       bash runtrain.sh
#     或者先修改脚本里 NODE_RANK=0, MASTER_ADDR=node0 后执行
#  2) 在其他从节点（node_rank=1,2,...）上分别执行：
#       NODE_RANK=1 bash runtrain.sh
#     或者直接改脚本中 NODE_RANK=1 后执行
#
# 如果使用 Slurm 或其他调度系统，可以根据实际需要：
#   srun -N2 -n2 --ntasks-per-node=1 bash runtrain.sh
# 这样调度系统会自动将脚本投递到两个节点上执行，node_rank 也可用环境变量注入
##############################################################################