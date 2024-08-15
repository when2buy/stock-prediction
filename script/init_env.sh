echo "download anaconda3..."
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.1-Linux-x86_64.sh \
  -O save/Anaconda3-5.3.1-Linux-x86_64.sh

echo "install anaconda3..."
bash save/Anaconda3-5.3.1-Linux-x86_64.sh

echo "create conda env..."
project_name="basename $(pwd)"
conda create -n "$project_name" python=3.7.13

echo "activate conda env..."
conda activate "$project_name"
