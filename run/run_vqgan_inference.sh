#!/bin/bash
JSONL_FILE="/mnt/bn/pretrain3d/real_word_data/preprocess/jsonls/dcr_data/2024_07_02-14_10_04-client_9a4bc499-3e43-49a5-a860-599d227b87ec.jsonl"
MODEL_WEIGHT="/mnt/bn/occupancy3d/workspace/lzy/Occ3d/work_dirs/pretrainv0.7_lsstpv_vits_multiextrin_datasetv0.2_rgb/epoch_1.pth"
QUANTIZER_TYPE="default"
TUNING_CODEBOOK=-1
LOCAL_EMBEDDING_PATH=""
USE_CBLINEAR=2
RATE_P=0.0
DISC_START=0
RATE_Q=1.0
RATE_D=1.0
MODE="eval"
GENERAL_MODE="vqgan"

VISUAL_DIR="./output/d508/t2/" 
VQVAE_WEIGHT="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/checkpoints_vqgan_16384_1600_mid1024/vqgan_epoch2_step9000.pth" 
MODEL="VAERes3DImgDirectBC"
INPUT_HEIGHT=518
INPUT_WIDTH=784
N_VISION_WORDS=16384
ENCODER_TYPE="vqgan"
VQ_CONFIG_PATH="/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/vqganlc/vqgan_configs/vqganlc_16384.yaml"
MID_CHANNELS=1024
E_DIM=1600

# 启动 Python 推理脚本
python3 inference.py \
    --visual_dir "${VISUAL_DIR}" \
    --jsonl_file "${JSONL_FILE}" \
    --model_weight "${MODEL_WEIGHT}" \
    --vqvae_weight "${VQVAE_WEIGHT}" \
    --input_height ${INPUT_HEIGHT} \
    --input_width ${INPUT_WIDTH} \
    --encoder_type "${ENCODER_TYPE}" \
    --quantizer_type "${QUANTIZER_TYPE}" \
    --tuning_codebook ${TUNING_CODEBOOK} \
    --n_vision_words ${N_VISION_WORDS} \
    --local_embedding_path "${LOCAL_EMBEDDING_PATH}" \
    --use_cblinear ${USE_CBLINEAR} \
    --rate_p ${RATE_P} \
    --disc_start ${DISC_START} \
    --rate_q ${RATE_Q} \
    --rate_d ${RATE_D} \
    --mode "${MODE}" \
    --model "${MODEL}" \
    --general_mode "${GENERAL_MODE}" \
    --vq_config_path "${VQ_CONFIG_PATH}" \
    --mid_channels ${MID_CHANNELS} \
    --e_dim ${E_DIM} \
    # --inp_channels=64 \
    # --out_channels=64 \

# choices: VAERes2DImgDirectBC, VQModel, Cog3DVAE
