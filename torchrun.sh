# sudo apt update
# sudo apt install libgl1-mesa-glx -y
# sudo apt install net-tools
# pip install torchdiffeq
# # pip install flow_matching
# # pip install torchmetrics
# pip install scipy
# pip install einops
# pip install matplotlib
# pip install opencv-python
# # pip install diffusers
# pip install omegaconf
# pip install numpy==1.26.4
# pip install debugpy
# pip install timm
# cd /mnt/bn/occupancy3d/workspace
# sudo chown -R tiger:tiger mzj
# chmod -R u+rwx mzj
# cd /mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/

# pip install thop
# pip install connected-components-3d

# ./TORCHRUN trainvae.py \
# --validate_path='./output/vae' \
# --batch_size=4 \
# --epochs=5 \
# --lr=2e-4 \
# --num_workers=8 \
# --mid_channels=320 \
# --checkpoint_dir="./checkpoints_vae" \
# --save_interval=2 \
# --val_interval=4 \
# --n_vision_words=1024 \
# --model="Cog3DVAE" \
# --mode="train" \
# --general_mode="vae" \
# --input_height=1380 \
# --input_width=1920 \
# --inp_channels=4 \
# --out_channels=4 \

python3 trainvae.py \
--validate_path='./output/vae' \
--batch_size=1 \
--epochs=5 \
--lr=2e-4 \
--num_workers=8 \
--mid_channels=320 \
--checkpoint_dir="./checkpoints_vae" \
--save_interval=1000 \
--val_interval=4000 \
--n_vision_words=1024 \
--model="Cog3DVAE" \
--mode="train" \
--general_mode="vae" \
--input_height=520 \
--input_width=784 \
--inp_channels=4 \
--out_channels=4 \