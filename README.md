To install

wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
export PATH="/home/hgupta/anaconda3/bin:$PATH"
conda create --name cryoem --file ./cryoemfinal/install.txt
conda activate cryoem
pip install "pillow<7"
