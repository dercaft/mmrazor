#
# create conda env
conda create -n wyh_create python=3.10
conda deactivate
conda activate wyh_create
# install pytorch 1.12 under CUDA 11.6
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# install opencv
pip install opencv-python-headless
# install mmcv-full
pip install mmcv-full==1.6.0
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html
pip install mmcls
pip install mmsegmentation
pip install mmdet
