sudo apt update
sudo apt install libgl1-mesa-glx -y
sudo apt install net-tools
pip install torchdiffeq
# pip install flow_matching
# pip install torchmetrics
pip install scipy
pip install einops
pip install matplotlib
pip install opencv-python
pip install omegaconf
pip install numpy==1.26.4
pip install debugpy
pip install timm
cd /mnt/bn/occupancy3d/workspace
sudo chown -R tiger:tiger mzj
chmod -R u+rwx mzj
cd /mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/

pip install thop
pip install connected-components-3d

/mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/TORCHRUN train.py \
--validate_path='./output/vqgan_16384_1600_32' \
--batch_size=1 \
--epochs=10 \
--lr=2e-4 \
--num_workers=8 \
--mid_channels=320 \
--inp_channels=64 \
--out_channels=64 \
--checkpoint_dir="./checkpoints_vqgan_16384_1600_32" \
--model="VAERes3DImgDirectBC" \
--save_interval=1000 \
--val_interval=4000 \
--n_vision_words=16384 \
--general_mode="vqgan" \