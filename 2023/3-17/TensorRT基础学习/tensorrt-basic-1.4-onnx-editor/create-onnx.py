import onnx # pip install onnx>=1.10.2
import onnx.helper as helper
import numpy as np

# https://github.com/onnx/onnx/blob/v1.2.1/onnx/onnx-ml.proto

nodes = [
    helper.make_node(
        name="Conv_0",   # 节点名字，不要和op_type搞混了
        op_type="Conv",  # 节点的算子类型, 比如'Conv'、'Relu'、'Add'这类，详细可以参考onnx给出的算子列表
        inputs=["image", "conv.weight", "conv.bias"],  # 各个输入的名字，结点的输入包含：输入和算子的权重。必有输入X和权重W，偏置B可以作为可选。
        outputs=["3"],  
        pads=[1, 1, 1, 1], # 其他字符串为节点的属性，attributes在官网被明确的给出了，标注了default的属性具备默认值。
        group=1,
        dilations=[1, 1],
        kernel_shape=[3, 3],
        strides=[1, 1]
    ),
    helper.make_node(
        name="ReLU_1",
        op_type="Relu",
        inputs=["3"],
        outputs=["output"]
    )
]

initializer = [
    helper.make_tensor(
        name="conv.weight",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1, 1, 3, 3],
        vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes(),
        raw=True
    ),
    helper.make_tensor(
        name="conv.bias",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1],
        vals=np.array([0.0], dtype=np.float32).tobytes(),
        raw=True
    )
]

inputs = [
    helper.make_value_info(
        name="image",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

outputs = [
    helper.make_value_info(
        name="output",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

graph = helper.make_graph(
    name="mymodel",
    inputs=inputs,
    outputs=outputs,
    nodes=nodes,
    initializer=initializer
)

# 如果名字不是ai.onnx，netron解析就不是太一样了
opset = [
    helper.make_operatorsetid("ai.onnx", 11)
]

# producer主要是保持和pytorch一致
model = helper.make_model(graph, opset_imports=opset, producer_name="pytorch", producer_version="1.9")
onnx.save_model(model, "my.onnx")

print(model)
print("Done.!")