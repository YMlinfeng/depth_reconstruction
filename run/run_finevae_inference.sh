#!/bin/bash
JSONL_FILE="/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec.jsonl"
MODEL_WEIGHT="/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth"
MODE="eval"
GENERAL_MODE="vae"

VISUAL_DIR="./output/d509/t1/" 
VQVAE_WEIGHT="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints_vqganlc_16384_2/vqgan_epoch3_step4000.pth" 
MODEL="VQModel"
INPUT_HEIGHT=518
INPUT_WIDTH=784
N_VISION_WORDS=16384
ENCODER_TYPE="vqgan_lc"
VQ_CONFIG_PATH="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/vqganlc/vqgan_configs/vqganlc_16384.yaml"
MID_CHANNELS=320
E_DIM=4

# 启动 Python 推理脚本
python3 inference.py \
    --visual_dir "${VISUAL_DIR}" \
    --jsonl_file "${JSONL_FILE}" \
    --model_weight "${MODEL_WEIGHT}" \
    --vqvae_weight "${VQVAE_WEIGHT}" \
    --input_height ${INPUT_HEIGHT} \
    --input_width ${INPUT_WIDTH} \
    --mid_channels ${MID_CHANNELS} \
    --e_dim ${E_DIM} \
    # --inp_channels=64 \
    # --out_channels=64 \

# choices: VAERes2DImgDirectBC, VQModel, Cog3DVAE
