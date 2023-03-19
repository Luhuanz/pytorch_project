# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
""" Utils to interact with the Triton Inference Server
"""
#Triton是NVIDIA开源的高性能推理服务器，可以用于快速部署和扩展机器学习模型的推理服务。
# 它支持多种框架和模型格式，可以通过gRPC或HTTP进行通信，并提供了灵活的扩展和自定义功能。
#Triton Inference Server是一个高性能开源推理服务器，可以用于推理深度学习模型。
import typing
from urllib.parse import urlparse

import torch


class TritonRemoteModel:
    """ A wrapper over a model served by the Triton Inference Server. It can
    be configured to communicate over GRPC or HTTP. It accepts Torch Tensors
    as input and returns them as outputs.
    """

    def __init__(self, url: str):
#ritonRemoteModel 是一个包装器，用于与 Triton 推理服务器上提供的模型进行交互。它可以配置为通过 GRPC 或 HTTP 进行通信。
# 它接受 Torch 张量作为输入，并将它们作为输出返回。
        """
        Keyword arguments:
        url: Fully qualified address of the Triton server - for e.g. grpc://localhost:8000
        """

        parsed_url = urlparse(url) #该函数首先解析 url，
# 判断使用 GRPC 还是 HTTP 通信，Python 的条件语句来判断 Triton 推理服务器的通信协议。
        if parsed_url.scheme == "grpc":
    #导入 tritonclient.grpc 模块中的 InferenceServerClient 和 InferInput 类。
            from tritonclient.grpc import InferenceServerClient, InferInput
#用于与其进行通信并进行推理任务。它有两个版本，一个是使用GRPC协议进行通信的版本，另一个是使用HTTP协议进行通信的版本。
            self.client = InferenceServerClient(parsed_url.netloc)  # Triton GRPC client
#model_repository 是获取 Triton Inference Server 上模型仓库的索引。
# 它包含有关仓库中可用模型的元数据信息，例如模型名称、版本等。这个信息会在后续的操作中使用到，例如获取模型元数据和输入/输出数据的形状、类型等。
            model_repository = self.client.get_model_repository_index()
#行代码是获取Triton推理服务器上的模型列表，然后从中获取第一个模型的名称，存储在TritonRemoteModel对象的self.model_name属性中。
            self.model_name = model_repository.models[0].name
#从 Triton 推理服务器中获取模型元数据（metadata）。self.model_name 存储了推理服务器上第一个模型的名称，它是通过 model_repository 获取的。as_json=True 参数用于将元数据以 JSON 格式返回。
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)
#该函数通过遍历 metadata['inputs'] 的每个元素，为每个输入占位符创建一个 InferInput 对象，
# 其中 i['name'] 是占位符的名称，[int(s) for s in i["shape"]] 是占位符的形状，i['datatype'] 是占位符的数据类型。
    # 最终，函数返回一个包含所有输入占位符的列表。
            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i['name'], [int(s) for s in i["shape"]], i['datatype']) for i in self.metadata['inputs']]

        else:
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton HTTP client
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository[0]['name']
            self.metadata = self.client.get_model_metadata(self.model_name)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i['name'], [int(s) for s in i["shape"]], i['datatype']) for i in self.metadata['inputs']]

        self._create_input_placeholders_fn = create_input_placeholders
#
    @property
#runtime 方法返回模型所在的运行时。在 Triton Inference Server 中，模型是由特定的后端（如 TensorFlow）或平台（如 Python）支持的，而这些后端或平台可以被称为模型的运行时。
    def runtime(self):
        """Returns the model runtime"""
        return self.metadata.get("backend", self.metadata.get("platform"))
#调用模型进行推理。它接受任意数量的位置参数和关键字参数。位置参数和关键字参数应该匹配模型的输入顺序或者输入名称。
    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        """ Invokes the model. Parameters can be provided via args or kwargs.
        args, if provided, are assumed to match the order of inputs of the model.
        kwargs are matched with the model input names.
        """
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata['outputs']:
            tensor = torch.as_tensor(response.as_numpy(output['name']))
            result.append(tensor)
        return result[0] if len(result) == 1 else result
#用于根据输入值创建 Triton 模型推理时所需的输入。
    def _create_inputs(self, *args, **kwargs):
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")

        placeholders = self._create_input_placeholders_fn()
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f"Expected {len(placeholders)} inputs, got {args_len}.")
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value.cpu().numpy())
        else:
            for input in placeholders:
                value = kwargs[input.name]
                input.set_data_from_numpy(value.cpu().numpy())
        return placeholders
