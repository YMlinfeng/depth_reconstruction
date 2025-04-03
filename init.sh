sudo apt update
sudo apt install libgl1-mesa-glx
sudo apt install net-tools
pip install torchdiffeq
# pip install flow_matching
# pip install torchmetrics
pip install scipy
pip install einops
pip install matplotlib
pip install opencv-python
# pip install diffusers
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


# pip install --upgrade pip setuptools
# pip install packaging
# pip install mmcv-full