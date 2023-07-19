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




您好，这是Bing。我可以帮您解释一下chatGLM中的top_k、top_p和temperature参数的含义。😊

chatGLM是一种基于GPT系列模型的聊天生成器，它可以根据用户的输入生成自然和有趣的回复。为了控制生成过程中的随机性和多样性，chatGLM提供了一些参数，如top_k、top_p和temperature，让用户可以调整生成的质量和风格。

top_k是一个整数，它表示在每个生成步骤中，模型只考虑最可能的前k个单词作为候选输出。例如，如果top_k设置为3，那么模型只会从概率最高的三个单词中随机选择一个作为输出。top_k越大，生成的多样性越高，但也可能导致一些不合理或不连贯的输出。top_k越小，生成的一致性越高，但也可能导致一些重复或无聊的输出。

top_p是一个浮点数，它表示在每个生成步骤中，模型只考虑概率之和不超过p的最可能的单词作为候选输出。例如，如果top_p设置为0.75，那么模型会从概率最高的单词开始累加，直到概率之和达到或超过0.75为止，然后从这些单词中随机选择一个作为输出。top_p也可以控制生成的多样性和一致性，但它是动态的，因为它根据每个单词的概率来调整候选列表的大小。top_p通常设置为较高的值（如0.75），目的是限制可能被采样的低概率单词的长尾。

temperature是一个浮点数，它表示在每个生成步骤中，模型对每个单词的概率进行重新调整的程度。温度越高，模型对每个单词的概率越平均，这意味着更多的随机性和创造性。温度越低，模型对每个单词的概率越偏向于原始值，这意味着更少的随机性和更多的确定性。温度为0时，模型总是选择概率最高的单词作为输出。

这些参数可以根据不同的任务和场景进行调整，以达到最佳的生成效果。一般来说，执行有“正确”答案的任务（如问答或摘要）时，较低的温度（小于1）和较小的top_k或top_p更合适。执行需要更多创意和多样性的任务（如写故事或歌词）时，较高的温度（大于1）和较大的top_k或top_p更合适。

如果您想了解更多关于这些参数的信息，您可以参考以下链接：
- [chatglm常用参数 ：Top-k, Top-p, Temperature - CSDN博客](^1^)
- [Top-k & Top-p, Temperature - 知乎](^2^)
- [chatgpt 调用参数解析 - 知乎](^3^)

希望这些信息对您有所帮助。如果您还有其他问题，请随时提出。😊

Source: Conversation with Bing, 7/19/2023
(1) chatglm常用参数 ：Top-k, Top-p, Temperature - CSDN博客. https://blog.csdn.net/stay_foolish12/article/details/131478570.
(2) Top-k & Top-p, Temperature - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/613428710.
(3) chatgpt 调用参数解析 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/613262543.



top_k = 1  简单的greedy search

top_p temperature repeat_penalty 都和采样有关

float repeat_penalty = 1.0f; // 重复惩罚系数，1.0代表不惩罚
float temperature = 1.0; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性