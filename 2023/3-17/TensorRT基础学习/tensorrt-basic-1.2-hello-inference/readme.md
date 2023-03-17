# 项目目标：学会使用tensorRT去推理构建好的模型
## 运行
```bash
make run
```

## 知识点
执行推理的步骤：
  1. 准备模型并加载
  2. 创建runtime：`createInferRuntime(logger)`
  3. 使用运行时时，以下步骤：
     1. 反序列化创建engine, 得为engine提供数据：`runtime->deserializeCudaEngine(modelData, modelSize)`,其中`modelData`包含的是input和output的名字，形状，大小和数据类型
        ```cpp
        class ModelData(object):
        INPUT_NAME = "data"
        INPUT_SHAPE = (1, 1, 28, 28) // [B, C, H, W]
        OUTPUT_NAME = "prob"
        OUTPUT_SIZE = 10
        DTYPE = trt.float32
        ```

     2. 从engine创建执行上下文:`engine->createExecutionContext()`
  4. 创建CUDA流`cudaStreamCreate(&stream)`：
     1. CUDA编程流是组织异步工作的一种方式，创建流来确定batch推理的独立
     2. 为每个独立batch使用IExecutionContext(3.2中已经创建了)，并为每个独立批次使用cudaStreamCreate创建CUDA流。
     
  5. 数据准备：
     1. 在host上声明`input`数据和`output`数组大小，搬运到gpu上
     2. 要执行inference，必须用一个指针数组指定`input`和`output`在gpu中的指针。
     3. 推理并将`output`搬运回CPU
  6. 启动所有工作后，与所有流同步以等待结果:`cudaStreamSynchronize`
  7. 按照与创建相反的顺序释放内存

