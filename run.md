```
docker run --gpus all \
-it --name trt \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v ${PWD}:/workspace/ \
nvcr.io/nvidia/pytorch:23.04-py3
```


```
docker run --gpus all \
-it --rm \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v ${PWD}:/home/faith/fastllm \
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

for win:
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64/:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:/home/faith/miniconda3/lib/python3.9/site-packages/nvidia/cublas/lib/


## optimization

https://zhuanlan.zhihu.com/p/620331719

https://zhuanlan.zhihu.com/p/619217800


compare time:
https://mdnice.com/writing/9f132e6af4694d0c9a6155d3cb31a8b4

torch.nn.functional.scaled_dot_product_attention

modeling_chatglm.py: CoreAttention


For CUDA tensor inputs, the function will dispatch into one of the following implementations:

1. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
2. Memory-Efficient Attention
3. A PyTorch implementation defined in C++

https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html


https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/fc133e4ffc6275f9d1c3a74ddd10e0a2/scaled_dot_product_attention_tutorial.ipynb


https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md


https://pytorch.org/blog/accelerating-large-language-models/


scaled_dot_product_attention 如何用 C++ 的 cuda 高效实现，这个问题没有一个确定的答案，不同的实现可能会有不同的优缺点。你可以参考一些已有的实现，比如：

- [PyTorch 中的 ScaledDotProduct.cpp](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/ScaledDotProduct.cpp)
- [FasterTransformer 中的 scaled_dot_product_attention.cu](https://github.com/NVIDIA/FasterTransformer/blob/main/src/decoding/scaled_dot_product_attention.cu)
- [PyTorch 的教程中的 (Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA)](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)

一般来说，高效实现 scaled_dot_product_attention 需要考虑以下几个方面：

- 利用 Tensor Core 的计算能力，使用 FP16 或者 INT8 的数据类型
- 合并或者融合一些操作，比如 bias fusion, layer normalization fusion, softmax fusion 等
- 优化内存使用和 IO 效率，比如 FlashAttention 和 Memory-Efficient Attention 等
- 根据输入数据和模型参数的不同，选择合适的算法和参数

Source: Conversation with Bing, 7/19/2023
(1) torch.nn.functional.scaled_dot_product_attention. https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html.
(2) (Beta) Implementing High-Performance Transformers with .... https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html.
(3) How to Implement Scaled Dot-Product Attention from Scratch .... https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/.


/home/faith/pytorch/aten/src/ATen/native/transformers/attention.cpp