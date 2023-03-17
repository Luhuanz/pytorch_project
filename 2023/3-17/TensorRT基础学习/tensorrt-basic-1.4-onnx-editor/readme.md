# 知识点
## 本项目抖音短视频辅助讲解
   - 0. 为什么要编辑onnx https://v.douyin.com/NPsJ4TV/
   - 1. pytorch生成onnx https://v.douyin.com/NPshBrF/
   - 2. 读取onnx https://v.douyin.com/NPsFELA/
   - 3. 编辑和创建onnx https://v.douyin.com/NPshVCL/！

## 本小节主要讲解onnx的原理，文件较多，我们一个一个慢慢来看：
1. `pytorch-gen-onnx.py`：是之前讲过的从pytorch转换onnx格式的代码。
2. 通过`onnx-ml.proto`和`make-onnx-pb.sh`了解onnx的结构
   - 2.1. onnx是基于protobuf来做数据存储和传输,*.proto后缀文件, 其定义是protobuf语法，类似json。
   - 2.2. 对于变量结构、类型等，我们可以参照`onnx-ml.proto`里面的定义。这个文件有800多行，放心我们只要搞清楚里面的核心部分就行：
       - `ModelProto`:当加载了一个onnx后，会获得一个`ModelProto`。它包含一个`GraphProto`和一些版本，生产者的信息。
      - `GraphProto`: 包含了四个repeated数组(可以用来存放N个相同类型的内容，key值为数字序列类型.)。这四个数组分别是node(`NodeProto`类型)，input(`ValueInfoProto`类型)，output(`ValueInfoProto`类型)和initializer(`TensorProto`类型)；
      - `NodeProto`: 存node，放了模型中所有的计算节点,语法结构如下：
        ![avatar](./figure/1.nodeproto.jpg)
      - `ValueInfoProto`: 存input，放了模型的输入节点。存output，放了模型中所有的输出节点；
      - `TensorProto`: 存initializer，放了模型的所有权重参数
      - `AttributeProto`:每个计算节点中还包含了一个`AttributeProto`数组，用来描述该节点的属性，比如Conv节点或者说卷积层的属性包含group，pad，strides等等；
   - 2.3. 通过protoc编译`onnx-ml.proto`，产生`onnx-ml.pb.cc`文件
        ```bash
        bash make-onnx-pb.sh
        ```
3.  create-onnx.py
       - 3.1. create-onnx.py直接从构建onnx，不经过任何框架的转换。通过import onnx和onnx.helper提供的make_node，make_graph，make_tensor等等接口我们可以轻易的完成一个ONNX模型的构建。
       - 3.2. 需要完成对node，initializer，input，output，graph，model的填充
       - 3.3. 读懂creat-onnx.py以make_node为例：
          ![avatar](./figure/2.make-node.jpg)
4.  edit-onnx.py
       - 4.1. 由于protobuf任何支持的语言，我们可以使用[c/c++/python/java/c#等等]实现对onnx文件的读写操作
       - 4.2. 掌握onnx和helper实现对onnx文件的各种编辑和修改
         - 增：一般伴随增加node和tensor
            ```python
            graph.initializer.append(xxx_tensor)
            graph.node.insert(0, xxx_node)
            ```
         - 删：
            ```python
            graph.node.remove(xxx_node)
            ```
         - 改：
             ```python
             input_node.name = 'data'
            ```
5.  read-onnx.py
    - 5.1 通过`graph`可以访问参数，数据是以protobuf的格式存储的，因此当中的数值会以bytes的类型保存。需要用`np.frombuffer`方法还原成类型为`float32`的`ndarray`。注意还原出来的`ndarray`是只读的。