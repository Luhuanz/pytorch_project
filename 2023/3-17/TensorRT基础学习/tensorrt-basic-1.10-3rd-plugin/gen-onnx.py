import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.autograd
import os
import json
from torch.onnx import register_custom_op_symbolic

class MYSELUImpl(torch.autograd.Function):

    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    @staticmethod
    def symbolic(g, x, p):
        print("==================================call symbolic")
        return g.op("Plugin", x, p, 
            g.op("Constant", value_t=torch.tensor([3, 2, 1], dtype=torch.float32)),
            name_s="MYSELU",
            info_s=json.dumps(
                dict(
                    attr1_s="这是字符串属性", 
                    attr2_i=[1, 2, 3], 
                    attr3_f=222
                ), ensure_ascii=False
            )
        )

    @staticmethod
    def forward(ctx, x, p):
        return x * 1 / (1 + torch.exp(-x))


class MYSELU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.param = nn.parameter.Parameter(torch.arange(n).float())

    def forward(self, x):
        return MYSELUImpl.apply(x, self.param)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.myselu = MYSELU(3)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.myselu(x)
        x[..., 0] = x[..., 0].sigmoid()
        return x


# 这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
# import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))

model = Model().eval()
input = torch.tensor([
    # batch 0
    [
        [1,   1,   1],
        [1,   1,   1],
        [1,   1,   1],
    ],
        # batch 1
    [
        [-1,   1,   1],
        [1,   0,   1],
        [1,   1,   -1]
    ]
], dtype=torch.float32).view(2, 1, 3, 3)

output = model(input)
print(f"inference output = \n{output}")

dummy = torch.zeros(1, 1, 3, 3)
torch.onnx.export(
    model, 

    # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
    (dummy,), 

    # 储存的文件路径
    "workspace/demo.onnx", 

    # 打印详细信息
    verbose=False, 

    # 为输入和输出节点指定名称，方便后面查看或者操作
    input_names=["image"], 
    output_names=["output"], 

    # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
    opset_version=11, 

    # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
    # 通常，我们只设置batch为动态，其他的避免动态
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },

    # 对于插件，需要禁用onnx检查
    enable_onnx_checker=False
)

import onnx
import onnx.helper as helper

model = onnx.load("workspace/demo.onnx")
for n in model.graph.node:
    if n.op_type == "ScatterND":
        n.op_type = "Plugin"
        n.attribute.append(helper.make_attribute("name", "MyScatterND"))
        n.attribute.append(helper.make_attribute("info", ""))

onnx.save(model, "workspace/demo.onnx")
print("Done.!")