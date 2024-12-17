#### Install
```
conda create -n textrans python=3.9

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c iopath iopath
conda install -c conda-forge python_abi
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu118.html
```