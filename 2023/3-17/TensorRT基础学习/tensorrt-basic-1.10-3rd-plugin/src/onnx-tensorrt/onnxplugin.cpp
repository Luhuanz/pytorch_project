
#include "onnxplugin.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>
#include <assert.h>
#include <string.h>

using namespace nvinfer1;
using namespace std;

namespace ONNXPlugin {

	GTensor::GTensor(float* ptr, int ndims, int* dims) {
		this->ptr_ = ptr;
		this->shape_.insert(shape_.end(), dims, dims + ndims);
		this->dtype_ = DataType::Float32;
	}

	int GTensor::offset_array(size_t size, const int* index_array) const{

		assert(size <= shape_.size());
		int value = 0;
		for(int i = 0; i < shape_.size(); ++i){

			if(i < size)
				value += index_array[i];

			if(i + 1 < shape_.size())
				value *= shape_[i+1];
		}
		return value;
	}

	int GTensor::offset_array(const std::vector<int>& index) const{
		return offset_array(index.size(), index.data());
	}

	GTensor::GTensor(float16* ptr, int ndims, int* dims) {
		this->ptr_ = ptr;
		this->shape_.insert(shape_.end(), dims, dims + ndims);
		this->dtype_ = DataType::Float16;
	}

	int GTensor::count(int start_axis) const {
		if(start_axis >= 0 && start_axis < shape_.size()){
			int size = 1;
			for (int i = start_axis; i < shape_.size(); ++i) 
				size *= shape_[i];
			return size;
		}else{
			return 0;
		}
	}

	/////////////////////////////////////////
	Weight::Weight(const std::vector<int>& dims, DataType dt){
		dims_ = dims;
		dt_ = dt;

		char* pshape = shape_string_;
		int volumn_size = 1;
		for(int i = 0; i < dims_.size(); ++i){
			volumn_size *= dims_[i];

			if(i + 1 < dims_.size()){
				pshape += sprintf(pshape, "%d x ", dims_[i]);
			}else{
				pshape += sprintf(pshape, "%d", dims_[i]);
			}
		}
		
		numel_ = volumn_size;
		data_bytes_ = DataTypeSizeOf(dt_) * numel_;
		cudaMallocHost(&pdata_host_, data_bytes_);
		cudaMalloc(&pdata_device_, data_bytes_);
	}

	void Weight::to_float32(){
		if(dt_ == DataType::Float32)
			return;

		if(dt_ != DataType::Float16){
			printf("Unsupport datatype %d to float32\n", dt_);
			return;
		}
		
		float* fp32ptr = nullptr;
		float* fp32device = nullptr;
		data_bytes_ = numel_ * sizeof(float);

		cudaMallocHost(&fp32ptr, data_bytes_);
		cudaMallocHost(&fp32device, data_bytes_);

		half* fp16ptr = static_cast<half*>(pdata_host_);
		for(int i = 0; i < numel_; ++i){
			fp32ptr[i] = __half2float(fp16ptr[i]);
		}
		free();

		cudaMemcpy(fp32device, fp32ptr, data_bytes_, cudaMemcpyHostToDevice);
		pdata_host_   = fp32ptr;
		pdata_device_ = fp32device;
	}

	void Weight::to_float16(){
		if(dt_ == DataType::Float16)
			return;

		if(dt_ != DataType::Float32){
			printf("Unsupport datatype %d to float16\n", dt_);
			return;
		}
		
		half* fp16ptr = nullptr;
		half* fp16device = nullptr;
		data_bytes_ = numel_ * sizeof(half);

		cudaMallocHost(&fp16ptr, data_bytes_);
		cudaMallocHost(&fp16device, data_bytes_);

		float* fp32ptr = static_cast<float*>(pdata_host_);
		for(int i = 0; i < numel_; ++i){
			fp16ptr[i] = __float2half(fp32ptr[i]);
		}
		free();

		cudaMemcpy(fp16device, fp16ptr, data_bytes_, cudaMemcpyHostToDevice);
		pdata_host_   = fp16ptr;
		pdata_device_ = fp16device;
	}

	void Weight::copy_to_gpu(){
		if(pdata_host_ == nullptr){
			printf("pdata_host_ is nullptr\n");
			return;
		}
		cudaMemcpy(pdata_device_, pdata_host_, data_bytes_, cudaMemcpyHostToDevice);
	}

	void Weight::free_host(){
		if(pdata_host_) cudaFreeHost(pdata_host_);
		pdata_host_ = nullptr;
	}

	void Weight::free(){
		if(pdata_host_) cudaFreeHost(pdata_host_);
		if(pdata_device_) cudaFree(pdata_device_);
		pdata_host_ = nullptr;
		pdata_device_ = nullptr;
	}
	/////////////////////////////////////
	InStream::InStream(const void* pdata, size_t size){
		pdata_ = static_cast<const unsigned char*>(pdata);
		size_ = size;
	}

	InStream& InStream::operator >> (std::string& value){
		int slen = 0;
		(*this) >> slen;
		value.resize(slen);
		read((char*)value.data(), slen);
		return *this;
	}
	
	void InStream::read(void* pdata, size_t size){
		if(cursor_ + size <= size_){
			memcpy(pdata, pdata_ + cursor_, size);
			cursor_ += size;
		}else{
			printf("Invalid read, cursor + size[%d] > total[%d]\n", cursor_ + size, size_);
		}
	}

	OutStream& OutStream::operator << (const char* value){
		int slen = strlen(value);
		(*this) << slen;
		write(value, slen);
		return *this;
	}

	OutStream& OutStream::operator << (const std::string& value){
		(*this) << value.c_str();
		return *this;
	}

	void OutStream::write(const void* pdata, size_t size){
		auto cp = static_cast<const unsigned char*>(pdata);
		data_.insert(data_.end(), cp, cp + size);
	}

	///////////////////////////////////
	LayerConfig::LayerConfig() {
		support_dtype_set_ = {nvinfer1::DataType::kFLOAT};
		support_plugin_format_set_ = {nvinfer1::PluginFormat::kLINEAR};
		usage_dtype_ = DataType::Float32;
		usage_plugin_format_ = nvinfer1::PluginFormat::kLINEAR;
	}

	void LayerConfig::serialize_data_copy_to(void* buffer) {
		if (!serialize_data_.empty())
			memcpy(buffer, &serialize_data_[0], serialize_data_.size());
	}

	int LayerConfig::serialize() {

		OutStream out;
		out << workspace_size_;
		out << usage_dtype_;
		out << max_batch_size_;
		out << usage_plugin_format_;
		out << info_;

		out << (int)weights_.size();
		for (int i = 0; i < weights_.size(); ++i) {

			if (usage_dtype_ == DataType::Float32) {
				weights_[i]->to_float32();
			}
			else if (usage_dtype_ == DataType::Float16) {
				weights_[i]->to_float16();
			}
			else{
				printf("unsupport datatype: %d\n", (int)usage_dtype_);
			}

			out << weights_[i]->dims_;
			out << weights_[i]->dt_;
			out.write((char*)weights_[i]->pdata_host_, weights_[i]->data_bytes_);
		}

		seril(out);
		serialize_data_ = out.data();
		return serialize_data_.size();
	}

	void LayerConfig::deserialize(const void* ptr, size_t length) {

		InStream in(ptr, length);
		in >> workspace_size_;
		in >> usage_dtype_;
		in >> max_batch_size_;
		in >> usage_plugin_format_;
		in >> info_;

		int nbWeights = 0;
		in >> nbWeights;

		weights_.resize(nbWeights);
		for (int i = 0; i < nbWeights; ++i) {
			std::vector<int> dims;
			in >> dims;

			DataType dt;
			in >> dt;

			weights_[i].reset(new Weight(dims, dt));
			in.read(weights_[i]->pdata_host_, weights_[i]->data_bytes_);
		}
		deseril(in);
	}

	void LayerConfig::setup(const std::string& info, const std::vector<std::shared_ptr<Weight>>& weights) {

		this->info_ = info;
		this->weights_ = weights;
	}

	///////////////////////////////////////////////////////////////////////////////////

	static DataType convert_trt_datatype(nvinfer1::DataType dt){
		switch(dt){
			case nvinfer1::DataType::kFLOAT: return DataType::Float32;
			case nvinfer1::DataType::kHALF: return DataType::Float16;
			case nvinfer1::DataType::kINT32: return DataType::Int32;
			default:
				printf("Unsupport data type %d\n", dt);
				return DataType::Float32;
		}
	}

	TRTPlugin::~TRTPlugin() {
	}

	void TRTPlugin::pluginInit(const std::string& name, const std::string& info, const std::vector<std::shared_ptr<Weight>>& weights) {
		phase_ = CompilePhase;
		layerName_ = name;
		config_ = this->new_config();
		config_->setup(info, weights);
		config_->init();
	}

	void TRTPlugin::pluginInit(const std::string& name, const void* serialData, size_t serialLength) {
		phase_ = InferencePhase;
		layerName_ = name;
		config_ = this->new_config();
		config_->deserialize(serialData, serialLength);
		config_->init();
	}

	std::shared_ptr<LayerConfig> TRTPlugin::new_config() {
		return std::shared_ptr<LayerConfig>(new LayerConfig());
	}

	int TRTPlugin::getNbOutputs() const noexcept{
		return config_->num_output_;
	}

	void TRTPlugin::configurePlugin(
		const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
		const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept{

		auto type = in->desc.type;
		auto format = in->desc.format;
		this->config_->usage_dtype_     = convert_trt_datatype(type);
		this->config_->usage_plugin_format_ = format;
		this->config_->num_input_ = nbInputs;
		this->config_->max_batch_size_ = in->max.d[0];
		this->config_finish();
	}

	int TRTPlugin::initialize() noexcept{
		return 0;
	}

	void TRTPlugin::terminate() noexcept{
	}

	bool TRTPlugin::supportsFormatCombination(
		int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept{
		
		bool match = config_->support_dtype_set_.find(inOut[pos].type) != config_->support_dtype_set_.end() &&
		config_->support_plugin_format_set_.find(inOut[pos].format) != config_->support_plugin_format_set_.end();
		return match;
	}

	size_t TRTPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,
		int32_t nbOutputs) const noexcept{
		return config_->workspace_size_;
	}

	int32_t TRTPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept{

		if (inputTensors_.empty()) {
			inputTensors_.resize(config_->num_input_);
			outputTensors_.resize(config_->num_output_);
			weightTensors_.resize(config_->weights_.size());

			for (int i = 0; i < weightTensors_.size(); ++i) {
				auto& w = config_->weights_[i];
				w->copy_to_gpu();
				w->free_host();

				weightTensors_[i].shape_ = w->dims_;
				weightTensors_[i].ptr_ = w->pdata_device_;
				weightTensors_[i].dtype_ = w->dt_;
			}
		}

		for (int i = 0; i < inputTensors_.size(); ++i) {
			inputTensors_[i].shape_ = std::vector<int>(inputDesc[i].dims.d, inputDesc[i].dims.d+inputDesc[i].dims.nbDims);
			inputTensors_[i].ptr_ = (void*)inputs[i];
			inputTensors_[i].dtype_ = convert_trt_datatype(inputDesc[i].type);
		}

		for (int i = 0; i < outputTensors_.size(); ++i) {
			outputTensors_[i].shape_ = std::vector<int>(outputDesc[i].dims.d, outputDesc[i].dims.d+outputDesc[i].dims.nbDims);
			outputTensors_[i].ptr_ = outputs[i];
			outputTensors_[i].dtype_ = convert_trt_datatype(outputDesc[i].type);
		}
		return enqueue(inputTensors_, outputTensors_, weightTensors_, workspace, stream);
	}

	size_t TRTPlugin::getSerializationSize() const noexcept{
		return config_->serialize();
	}

	void TRTPlugin::serialize(void* buffer) const noexcept{
		config_->serialize_data_copy_to(buffer);
	}
};// namespace Plugin