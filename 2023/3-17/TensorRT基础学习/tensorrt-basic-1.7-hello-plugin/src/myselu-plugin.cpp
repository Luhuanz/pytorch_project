#include "myselu-plugin.hpp"
#include "NvInfer.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

void myselu_inference(const float* x, float* output, int n, cudaStream_t stream);

// MySELU plugin的特定常量
namespace
{
const char* MYSELU_PLUGIN_VERSION{"1"};  // 采用的名称要对应上onnx-tensorrt-release-8.0/builtin_op_importers.cpp:5094行定义的名称
const char* MYSELU_PLUGIN_NAME{"MYSELU"};
} // namespace

// 静态类字段的初始化
PluginFieldCollection MySELUPluginCreator::mFC{}; // FieldCollection 字段收集
std::vector<PluginField> MySELUPluginCreator::mPluginAttributes;
// 实际注册时，注册的是创建器，交给tensorRT管理
REGISTER_TENSORRT_PLUGIN(MySELUPluginCreator);

// 用于序列化插件的Helper function
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// 用于反序列化插件的Helper function
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
// 定义插件类MYSELUPlugin
MySELUPlugin::MySELUPlugin(const std::string name, const std::string attr1, float attr3)
    : mLayerName(name),
    mattr1(attr1),
    mattr3(attr3)
{
    printf("==================== 编译阶段，attr1 = %s, attr3 = %f\n", attr1.c_str(), attr3);
}

MySELUPlugin::MySELUPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    int nstr = readFromBuffer<int>(d);
    mattr1 = std::string(d, d + nstr);

    d += nstr;
    mattr3 = readFromBuffer<float>(d);
    assert(d == (a + length));

    printf("==================== 推理阶段，attr1 = %s, attr3 = %f\n", mattr1.c_str(), mattr3);
}

const char* MySELUPlugin::getPluginType() const noexcept
{
    return MYSELU_PLUGIN_NAME;
}

const char* MySELUPlugin::getPluginVersion() const noexcept
{
    return MYSELU_PLUGIN_VERSION;
}

int MySELUPlugin::getNbOutputs() const noexcept
{
    return 1;
}
// 获取该层的输出维度是多少
nvinfer1::DimsExprs MySELUPlugin::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Validate input arguments
    // assert(nbInputDims == 1);

    // MySELUping不改变输入尺寸，所以输出尺寸将与输入尺寸相同
    return *inputs;
}

int MySELUPlugin::initialize() noexcept
{
    return 0;
}
// 行性能测试
int MySELUPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    void* output = outputs[0];
    size_t volume = 1;
    for (int i = 0; i < inputDesc->dims.nbDims; i++){
        volume *= inputDesc->dims.d[i];
    }
    mInputVolume = volume;

    myselu_inference(
        static_cast<const float*>(inputs[0]), 
        static_cast<float*>(output), 
        mInputVolume,
        stream
    );
    return 0;
}

size_t MySELUPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) + mattr1.size() + sizeof(mattr3);
}
// 该层的参数序列化储存为trtmodel文件
void MySELUPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    int nstr = mattr1.size();
    writeToBuffer(d, nstr);
    memcpy(d, mattr1.data(), nstr);

    d += nstr;
    writeToBuffer(d, mattr3);

    assert(d == a + getSerializationSize());
}
// 判断该插件所支持的数据格式和类型
bool MySELUPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{   
    auto type = inOut[pos].type;
    auto format = inOut[pos].format;
    // 这个插件只支持普通的浮点数，以及NCHW输入格式
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
        return true;
    else
        return false;
}

void MySELUPlugin::terminate() noexcept {}

void MySELUPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}
// 配置插件格式:告诉你目前这个层所采用的数据格式和类型
void MySELUPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
    const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept{
    // Validate input arguments

    auto type = in->desc.type;
    auto format = in->desc.format;
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kLINEAR);
}
// 克隆插件
IPluginV2DynamicExt* MySELUPlugin::clone() const noexcept
{
    printf("===================克隆插件=================\n");
    auto plugin = new MySELUPlugin(mLayerName, mattr1, mattr3);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void MySELUPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MySELUPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
// 插件创建器
MySELUPluginCreator::MySELUPluginCreator()
{
    // 描述MySELUPlugin的必要PluginField参数
    mPluginAttributes.emplace_back(PluginField("attr1", nullptr, PluginFieldType::kCHAR, 0));
    mPluginAttributes.emplace_back(PluginField("attr3", nullptr, PluginFieldType::kFLOAT32, 1));

    // 收集PluginField的参数
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MySELUPluginCreator::getPluginName() const noexcept
{
    return MYSELU_PLUGIN_NAME;
}

const char* MySELUPluginCreator::getPluginVersion() const noexcept
{
    return MYSELU_PLUGIN_VERSION;
}

const PluginFieldCollection* MySELUPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
// 创建plugin
IPluginV2* MySELUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    std::string attr1;
    float attr3;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    for (int i = 0; i < fc->nbFields; i++)
    {
        if (strcmp(fields[i].name, "attr1") == 0)
        {
            assert(fields[i].type == PluginFieldType::kCHAR);
            auto cp = static_cast<const char*>(fields[i].data);
            attr1 = std::string(cp, cp + fields[i].length);
        }
        else if (strcmp(fields[i].name, "attr3") == 0)
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            attr3 = *(static_cast<const float*>(fields[i].data));
        }
    }
    return new MySELUPlugin(name, attr1, attr3);
}
// 反序列化插件参数进行创建
IPluginV2* MySELUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call MySELUPlugin::destroy()
    return new MySELUPlugin(name, serialData, serialLength);
}

void MySELUPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MySELUPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}