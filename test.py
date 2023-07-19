
from transformers import AutoTokenizer, AutoModel
from fastllm_pytools import llm
tokenizer = AutoTokenizer.from_pretrained("chatglm2_6b", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/home/faith/fastllm/chatglm2-6b-int4", trust_remote_code=True)
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model = AutoModel.from_pretrained("/home/faith/chatglm2-6b", trust_remote_code=True)
# model = llm.from_hf(model, tokenizer, dtype="float16")  # 可以调整为"float16", "int8", "int4"


model = llm.model("chatglm2-6b-fp16.flm")
# model = llm.model("/home/faith/fastllm/chatglm2-6b-int4.flm")
start = time.time()
text = "你好,请帮我计算一个数学题:两个苹果加上六个苹果,等于多少个苹果"
text = '''已知信息：
    以下内容都是提问的设备3045的相关信息:15.照明装置设备整体结构大概可分为 部分组成整机重量 吨为方便设备运输或移动给料斗可升降尾料输送机可折叠侧面楼梯可折叠本章只记录主要零部件如想查阅更详细的零部件信息请参考KJ-3045B 机架图片的链接是: http://img.com/cf9e206b436a624a0f5f89e6f24eab16.png-3序号 10按键可以控制柴油机的速度4.03 设备形态转换当设备移动到指定地点后便可从运输状态图 4.03-1转换为工作状态图 57 -图片的链接是: http://img.com/273bdbe9b0ad0dd0fe687a4c1d0b3261.png 图 4.03-1 设备运输状态图片的链接是: http://img.com/2bf1e565bf2f16f849f65458600ee3fa.png 图 4.03-2 设备工作状态 58 - 尾料输送机工作状态/运 输状态装换一拆除运输固定装置拆除尾料输送机运输固定螺栓图 -1序号 1图片的链接是: http://img.com/baab6a53793c8430be0b03272ae3816b.png 1.尾料输送机运输固定螺栓图 -1 尾料输送机运输状态二展开尾料输送机1.展开 尾料输送机一段操作遥控器将尾料输送机一段摇杆图 -3缓缓下拨此时尾料输送机一段油7才能使柴油机停止工作手动模式下遥控器无法控制柴油机停止自动模式下只需要按一下遥控器上的柴油机停止按钮-3序号 8柴油机就会减速直至停止如需要更详细的介绍请参照柴油机控制柜使用说明书4.04 设备安装设备安装只能由我公司售后服务工程师或受过专业操作技能培训的人员获得用户授权后方可进行安装作业 

    根据上述已知信息，简洁和专业回答用户的问题，如果问题里询问图片，请返回相关的图片具体链接。如果已知信息里有多个图片，请返回最匹配的图片链接，并用[]包含链接内容而且不要有其他文字描述。
    如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：3045的运输状态图片'''
outs = ""
history = [['你好', '你好，我是机器人']]
for i in range(1):
    _start = time.time()
    out = model.response(text, do_sample=False, temperature=1.0, history=history)
    outs += out

    print("throughput: {} {}".format((len(tokenizer.tokenize(out)) / (time.time() - _start)), "tokens/s"))
print(outs)
end = time.time()
token_count = len(tokenizer.tokenize(outs))
print('\ngenerate token number', token_count, 'time consume', end - start, 's')
print((end - start) * 1000 / token_count, 'ms/token')
print("throughput: {} {}".format((token_count / (end - start)), "tokens/s"))




# '[Round 1]\n\n问：你好\n\n答：你好，我是机器人\n\n[Round 2]\n\n问：已知信息：\n    以下内容都是提问的设备3045的相关信息:15.照明装置设备整体结构大概可分为 部分组成整机重量 吨为方便设备运输或移动给料斗可升降尾料输送机可折叠侧面楼梯可折叠本章只记录主要零部件如想查阅更详细的零部件信息请参考KJ-3045B 机架图片的链接是: http://img.com/cf9e206b436a624a0f5f89e6f24eab16.png-3序号 10按键可以控制柴油机的速度4.03 设备形态转换当设备移动到指定地点后便可从运输状态图 4.03-1转换为工作状态图 57 -图片的链接是: http://img.com/273bdbe9b0ad0dd0fe687a4c1d0b3261.png 图 4.03-1 设备运输状态图片的链接是: http://img.com/2bf1e565bf2f16f849f65458600ee3fa.png 图 4.03-2 设备工作状态 58 - 尾料输送机工作状态/运输状态装换一拆除运输固定装置拆除尾料输送机运输固定螺栓图 -1序号 1图片的链接是: http://img.com/baab6a53793c8430be0b03272ae3816b.png 1.尾料输送机运输固定螺栓图 -1 尾料输送机运输状态二展开尾料输送机1.展开尾料输送机一段操作遥控器将尾料输送机一段摇杆图 -3缓缓下拨此时尾料输送机一段油7才能使柴油机停止工作手动模式下遥控器无法控制柴油机停止自动模式下只需要按一下遥控器上的柴油机停止按钮-3序号 8柴油机就会减速直至停止如需要更详细的介绍请参照柴油机控制柜使用说明书4.04 设备安装设备安装只能由我公司售后服务工程师或受过专业操作技能培训的人员获得用户授权后方可进行安装作业 \n\n    根据上述已知信息，简洁和专业回答用户的问题，如果问题里询问图片，请返回相关的图片具体链接。如果已知信息里有多个图片，请返回最匹配的图片链接，并用[]包含链接内容而且不要有其他文字描述。\n    如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：3045的运输状态图片\n\n答：'

# [Round 0]
# 问：你好
# 答：你好，我是机器人
# [Round 1]
# 问：已知信息：
#     以下内容都是提问的设备3045的相关信息:15.照明装置设备整体结构大概可分为 部分组成整机重量 吨为方便设备运输或移动给料斗可升降尾料输送机可折叠侧面楼梯可折叠本章只记录主要零部件如想查阅更详细的零部件信息请参考KJ-3045B 机架图片的链接是: http://img.com/cf9e206b436a624a0f5f89e6f24eab16.png-3序号 10按键可以控制柴油机的速度4.03 设备形态转换当设备移动到指定地点后便可从运输状态图 4.03-1转换为工作状态图 57 -图片的链接是: http://img.com/273bdbe9b0ad0dd0fe687a4c1d0b3261.png 图 4.03-1 设备运输状态图片的链接是: http://img.com/2bf1e565bf2f16f849f65458600ee3fa.png 图 4.03-2 设备工作状态 58 - 尾料输送机工作状态/运 输状态装换一拆除运输固定装置拆除尾料输送机运输固定螺栓图 -1序号 1图片的链接是: http://img.com/baab6a53793c8430be0b03272ae3816b.png 1.尾料输送机运输固定螺栓图 -1 尾料输送机运输状态二展开尾料输送机1.展开 尾料输送机一段操作遥控器将尾料输送机一段摇杆图 -3缓缓下拨此时尾料输送机一段油7才能使柴油机停止工作手动模式下遥控器无法控制柴油机停止自动模式下只需要按一下遥控器上的柴油机停止按钮-3序号 8柴油机就会减速直至停止如需要更详细的介绍请参照柴油机控制柜使用说明书4.04 设备安装设备安装只能由我公司售后服务工程师或受过专业操作技能培训的人员获得用户授权后方可进行安装作业 

#     根据上述已知信息，简洁和专业回答用户的问题，如果问题里询问图片，请返回相关的图片具体链接。如果已知信息里有多个图片，请返回最匹配的图片链接，并用[]包含链接内容而且不要有其他文字描述。
#     如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：3045的运输状态图片
# 答：