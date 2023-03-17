
#ifndef ONNX_PLUGIN_HPP
#define ONNX_PLUGIN_HPP

#include <memory>
#include <vector>
#include <set>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>

namespace ONNXPlugin {

	enum Phase {
		CompilePhase,
		InferencePhase
	};

	typedef unsigned short float16;

	enum class DataType : int {
        Float32 = 0,
        Float16 = 1,
        Int32 = 2,
        UInt8 = 3
    };

	inline int DataTypeSizeOf(DataType dt){
		switch(dt){
			case DataType::Float32: return 4;
			case DataType::Float16: return 2;
			case DataType::Int32: return 4;
			case DataType::UInt8: return 1;
			default: return 0;
		}
	}

	inline const char* data_type_string(DataType dt){
		switch(dt){
			case DataType::Float32: return "Float32";
			case DataType::Float16: return "Float16";
			case DataType::Int32: return "Int32";
			case DataType::UInt8: return "UInt8";
			default: return "UnknowDataType";
		}
	}

	struct GTensor {
		GTensor() {}
		GTensor(float* ptr, int ndims, int* dims);
		GTensor(float16* ptr, int ndims, int* dims);

		int count(int start_axis = 0) const;

		template<typename ... _Args>
		int offset(int index, _Args&& ... index_args) const{
			const int index_array[] = {index, index_args...};
            return offset_array(sizeof...(index_args) + 1, index_array);
		}

		int offset_array(const std::vector<int>& index) const;
		int offset_array(size_t size, const int* index_array) const;

		inline int batch()   const{return shape_[0];}
        inline int channel() const{return shape_[1];}
        inline int height()  const{return shape_[2];}
        inline int width()   const{return shape_[3];}

		template<typename _T>
		inline _T* ptr() const { return (_T*)ptr_; }

		template<typename _T, typename ... _Args>
		inline _T* ptr(int i, _Args&& ... args) const { return (_T*)ptr_ + offset(i, args...); }

		void* ptr_ = nullptr;
		DataType dtype_ = DataType::Float32;
		std::vector<int> shape_;
	};

	struct Weight{
		Weight() = default;
		Weight(const std::vector<int>& dims, DataType dt);
		void to_float32();
		void to_float16();
		void copy_to_gpu();
		void free_host();
		void free();

		void* pdata_host_ = nullptr;
		void* pdata_device_ = nullptr;
		size_t data_bytes_ = 0;
		size_t numel_ = 0;
		std::vector<int> dims_;
		DataType dt_ = DataType::Float32;
		char shape_string_[100] = {0};
	};

	class InStream{
	public:
		InStream(const void* pdata, size_t size);
		InStream& operator >> (std::string& value);

		template<typename _T>
		InStream& operator >> (std::vector<_T>& value){
			int size = 0;
			(*this) >> size;
			value.resize(size);
			for(int i = 0; i < size; ++i)
				(*this) >> value[i];
			return *this;
		}

		template<typename _T>
		InStream& operator >> (_T& value){
			read(&value, sizeof(_T));
			return *this;
		}

		void read(void* pdata, size_t size);
		int cursor(){return cursor_;}

	private:
		const unsigned char* pdata_ = nullptr;
		size_t size_ = 0;
		int cursor_ = 0;
	};

	class OutStream{
	public:
		OutStream& operator << (const char* value);
		OutStream& operator << (const std::string& value);

		template<typename _T>
		OutStream& operator << (const std::vector<_T>& value){
			int size = value.size();
			(*this) << size;
			for(int i = 0; i < size; ++i)
				(*this) << value[i];
			return *this;
		}

		template<typename _T>
		OutStream& operator << (const _T& value){
			write(&value, sizeof(_T));
			return *this;
		}

		void write(const void* pdata, size_t size);
		const std::vector<unsigned char>& data(){return data_;}

	private:
		std::vector<unsigned char> data_;
	};

	struct LayerConfig {

		///////////////////////////////////
		int num_output_ = 1;
		int num_input_  = 1;
		size_t workspace_size_ = 0;
		int max_batch_size_ = 0;
		std::set<nvinfer1::DataType> support_dtype_set_;
		std::set<nvinfer1::PluginFormat> support_plugin_format_set_;

		std::vector<std::shared_ptr<Weight>> weights_;
		DataType usage_dtype_;
		nvinfer1::PluginFormat usage_plugin_format_;
		std::string info_;

		///////////////////////////////////
		std::vector<unsigned char> serialize_data_;

		LayerConfig();
		void serialize_data_copy_to(void* buffer);
		int serialize();
		void deserialize(const void* ptr, size_t length);
		void setup(const std::string& info, const std::vector<std::shared_ptr<Weight>>& weights);
		virtual void seril(OutStream& out) {}
		virtual void deseril(InStream& in) {}
		virtual void init(){}
	};

	#define SetupPlugin(class_)			\
		virtual const char* getPluginType() const noexcept override{return #class_;};																		\
		virtual const char* getPluginVersion() const noexcept override{return "1";};																			\
		virtual nvinfer1::IPluginV2DynamicExt* clone() const noexcept override{return new class_(*this);}

	#define RegisterPlugin(class_)		\
	class class_##PluginCreator__ : public nvinfer1::IPluginCreator{																				\
	public:																																			\
		const char* getPluginName() const noexcept override{return #class_;}																					\
		const char* getPluginVersion() const noexcept override{return "1";}																					\
		const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override{return &mFieldCollection;}													\
																																					\
		nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override{									\
			auto plugin = new class_();																												\
			mFieldCollection = *fc;																													\
			mPluginName = name;																														\
			return plugin;																															\
		}																																			\
																																					\
		nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override{								\
			auto plugin = new class_();																												\
			plugin->pluginInit(name, serialData, serialLength);																						\
			mPluginName = name;																														\
			return plugin;																															\
		}																																			\
																																					\
		void setPluginNamespace(const char* libNamespace) noexcept override{mNamespace = libNamespace;}														\
		const char* getPluginNamespace() const noexcept override{return mNamespace.c_str();}																	\
																																					\
	private:																																		\
		std::string mNamespace;																														\
		std::string mPluginName;																													\
		nvinfer1::PluginFieldCollection mFieldCollection{0, nullptr};																				\
	};																																				\
	REGISTER_TENSORRT_PLUGIN(class_##PluginCreator__);

	class TRTPlugin : public nvinfer1::IPluginV2DynamicExt {
	public:
		virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override{return inputTypes[0];}

		virtual void configurePlugin(
			const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
			const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;

		virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) noexcept override {}
		virtual void detachFromContext() noexcept override {}
		virtual void setPluginNamespace(const char* pluginNamespace) noexcept override{this->namespace_ = pluginNamespace;};
		virtual const char* getPluginNamespace() const noexcept override{return this->namespace_.data();};

		virtual ~TRTPlugin();
		virtual int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) = 0;

		void pluginInit(const std::string& name, const std::string& info, const std::vector<std::shared_ptr<Weight>>& weights);
		void pluginInit(const std::string& name, const void* serialData, size_t serialLength);
		virtual void config_finish() {};

		virtual std::shared_ptr<LayerConfig> new_config();
		virtual bool supportsFormatCombination(
			int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

		virtual int getNbOutputs() const noexcept;
		virtual nvinfer1::DimsExprs getOutputDimensions(
        	int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept{return inputs[0];}

		virtual int initialize() noexcept;
		virtual void terminate() noexcept;
		virtual void destroy() noexcept override{}
		virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,
        	int32_t nbOutputs) const noexcept override;

		virtual int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
            const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

		virtual size_t getSerializationSize() const noexcept override;
		virtual void serialize(void* buffer) const noexcept override;

	protected:
		std::string namespace_;
		std::string layerName_;
		Phase phase_ = CompilePhase;
		std::shared_ptr<LayerConfig> config_;
		std::vector<GTensor> inputTensors_;
		std::vector<GTensor> outputTensors_;
		std::vector<GTensor> weightTensors_;
	};

}; //namespace Plugin

#endif //ONNX_PLUGIN_HPP