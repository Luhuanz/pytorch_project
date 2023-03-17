import onnx
import onnx.helper as helper
import numpy as np

model = onnx.load("demo.onnx")

# 可以取出权重
conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]
# 修改权
conv_weight.raw_data = np.arange(9, dtype=np.float32).tobytes()

# 修改权重后储存
onnx.save_model(model, "demo.change.onnx")
print("Done.!")