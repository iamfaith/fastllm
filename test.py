
from transformers import AutoTokenizer, AutoModel
from fastllm_pytools import llm
tokenizer = AutoTokenizer.from_pretrained("chatglm2_6b", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/home/faith/chatglm2-6b", trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained("/home/faith/fastllm/chatglm2-6b-int4", trust_remote_code=True)
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# model = AutoModel.from_pretrained("/home/faith/chatglm2-6b", trust_remote_code=True)
# model = llm.from_hf(model, tokenizer, dtype="float16")  # 可以调整为"float16", "int8", "int4"


model = llm.model("chatglm2-6b-fp16.flm")
# model = llm.model("/home/faith/fastllm/chatglm2-6b-int4.flm")
start = time.time()
text = "你好,请帮我计算一个数学题:两个苹果加上六个苹果,等于多少个苹果"
outs = ""
for i in range(8):
    _start = time.time()
    out = model.response(text, do_sample=False)
    outs += out

    print("throughput: {} {}".format((len(tokenizer.tokenize(out)) / (time.time() - _start)), "tokens/s"))
print(outs)
end = time.time()
token_count = len(tokenizer.tokenize(outs))
print('\ngenerate token number', token_count, 'time consume', end - start, 's')
print((end - start) * 1000 / token_count, 'ms/token')
print("throughput: {} {}".format((token_count / (end - start)), "tokens/s"))
