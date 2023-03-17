1. cuda 安装

2. cudnn 安装  

   `tar -xzvf cudnn-8.0.5-linux-x64-cuda11.0.tgz `  解压

      sudo cp cuda/include/cudnn*.h /usr/local/cuda/include/
      sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
      sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
3. 安装Tensorrt





