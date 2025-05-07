# sudo apt update
# sudo apt install libgl1-mesa-glx -y
# sudo apt install net-tools
# pip install torchdiffeq
# pip install flow_matching
# pip install torchmetrics
# pip install scipy
# pip install einops
# pip install matplotlib
# pip install opencv-python
# pip install omegaconf
# pip install numpy==1.26.4
# pip install debugpy
# pip install timm
# pip install clip
# pip install albumentations
# cd /mnt/bn/occupancy3d/workspace
# sudo chown -R tiger:tiger mzj
# chmod -R u+rwx mzj
# cd /mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/

# pip install thop
# pip install connected-components-3d
# pip install imageio
# pip install accelerate
# pip install imageio[ffmpeg]

bash /mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/TORCHRUN trainvae.py \
--validate_path='./output/vae_c3_17' \
--batch_size=1 \
--epochs=10 \
--lr=2e-4 \
--num_workers=8 \
--mid_channels=1024 \
--checkpoint_dir="./checkpoints_vae_c3_17" \
--save_interval=1000 \
--val_interval=2000 \
--n_vision_words=1024 \
--model="Cog3DVAE" \
--mode="train" \
--input_height=520 \
--input_width=784 \
--inp_channels=3 \
--out_channels=3 \
# --concat \





