```
docker run --gpus all \
-it --name trt \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v ${PWD}:/workspace/ \
nvcr.io/nvidia/pytorch:23.04-py3
```


``` sh
cd fastllm
mkdir build
cd build
cmake .. -DUSE_CUDA=ON # 如果不使用GPU编译，那么使用 cmake .. -DUSE_CUDA=OFF
make -j16
cd tools && python setup.py install
```


export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64/:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:/home/faith/miniconda3/lib/python3.9/site-packages/nvidia/cublas/lib/