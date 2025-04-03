#imagenet_path="/mnt/bn/robotics-mllm-lq/mingyao/workspace_tiger/VQGAN-LC/blob/color/1724732978518067968.jpg"
imagenet_path="/mnt/bn/robotics-mllm-lq/mingyao/workspace_tiger/VQGAN-LC/blob/color/1724733051411238400.jpg"
codebook_path="/mnt/bn/robotics-mllm-lq/mingyao/workspace_tiger/VQGAN-LC/blob/codebook/codebook-100K.pth"
vq_path="/mnt/bn/robotics-mllm-lq/mingyao/workspace_tiger/VQGAN-LC/blob/vqganlc/vqgan-lc-100K-f16-dim8.pth"
#torchrun --nproc_per_node 1 eval_generation.py \
python reconstruct.py \
    --image_size  1920\
    --imagenet_path $imagenet_path \
    --vq_config_path vqgan_configs/vq-f16.yaml \
    --output_dir "log_eval_gpt/foo-vq-f16" \
    --quantizer_type "org" \
    --local_embedding_path $codebook_path \
    --stage_1_ckpt $vq_path \
    --tuning_codebook 0 \
    --embed_dim 8 \
    --n_vision_words 100000 \
    --use_cblinear 1 \