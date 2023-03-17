# 知识点
1. 插件主要是继承自IPluginV2DynamicExt后实现特定接口即可

# 实现插件要点
1. 导出onnx的时候，为module增加symbolic函数
    - 参照这里：https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    - g.op对应的名称，需要与下面解析器的名称对应
2. src/onnx-tensorrt-release-8.0/builtin_op_importers.cpp:5094行，添加对插件op的解析
    - DEFINE_BUILTIN_OP_IMPORTER(MYSELU)
    - 注意解析时采用的名称要匹配上src/myselu-plugin.cpp:15行
3. src/myselu-plugin.cpp:183行，创建MySELUPluginCreator，插件创建器
    - 实际注册时，注册的是创建器，交给tensorRT管理
    - REGISTER_TENSORRT_PLUGIN(MySELUPluginCreator);
    - src/myselu-plugin.cpp:23行
4. src/myselu-plugin.cpp:42行，定义插件类MySELUPlugin
    - Creator创建器来实例化MySELUPlugin类
5. 正常使用该onnx即可

# 插件的阶段phase
1. 编译阶段
    - 1. 通过MySELUPluginCreator::createPlugin创建plugin
    - 2. 期间会调用MySELUPlugin::clone克隆插件
    - 3. 调用MySELUPlugin::supportsFormatCombination判断该插件所支持的数据格式和类型
        - 在这里我们告诉引擎，本插件可以支持什么类型的推理
        - 可以支持多种，例如fp32、fp16、int8等等
    - 4. 调用MySELUPlugin::getOutputDimensions获取该层的输出维度是多少
    - 5. 调用MySELUPlugin::enqueue进行性能测试（不是一定会执行）
        - 如果支持多种，则会在多种里面进行实际测试，选择一个性能最好的配置
    - 6. 调用MySELUPlugin::configurePlugin配置插件格式
        - 告诉你目前这个层所采用的数据格式和类型
    - 7. 调用MySELUPlugin::serialize将该层的参数序列化储存为trtmodel文件
2. 推理阶段
    - 1. 通过MySELUPluginCreator::deserializePlugin反序列化插件参数进行创建
    - 2. 期间会调用MySELUPlugin::clone克隆插件
    - 3. 调用MySELUPlugin::configurePlugin配置当前插件使用的数据类型和格式
    - 4. 调用MySELUPlugin::enqueue进行推理