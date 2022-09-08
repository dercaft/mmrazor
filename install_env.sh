#
# create conda env
conda create -n wyh_create python=3.10
conda deactivate
conda activate wyh_create
# install pytorch 1.12 under CUDA 11.6
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# install opencv
# install mmcv-full
pip install mmcv-full
pip install mmcls
pip install mmsegmentation
pip install mmdet
