sudo apt update
sudo apt install libgl1-mesa-glx -y
sudo apt install net-tools
pip install torchdiffeq
pip install flow_matching
pip install torchmetrics
pip install scipy
pip install einops
pip install matplotlib
pip install opencv-python
pip install omegaconf
pip install numpy==1.26.4
pip install debugpy
pip install timm
pip install clip
pip install albumentations
cd /mnt/bn/occupancy3d/workspace
sudo chown -R tiger:tiger mzj
chmod -R u+rwx mzj
cd /mnt/bn/occupancy3d/workspace/mzj/mp_pretrain/

pip install thop
pip install connected-components-3d
pip install imageio
pip install accelerate
pip install imageio[ffmpeg]
pip install pytorch-lightning


# pip install --upgrade pip setuptools
# pip install packaging
# pip install mmcv-full