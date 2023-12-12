import inspect
import sys

from kserve import InferRequest

from enki_client import request_types, response_types

def list_models():
    current_module = sys.modules[__name__]
    classes = []
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and issubclass(obj, Model) and obj != Model:
            classes.append(obj.__name__)
    return classes

class Model():
    def __init__(self, name, input_type, output_type, endpoint):
        self.name = name
        self.input_type = input_type
        self.output_type = output_type
        self.endpoint = endpoint

    def to_inference_request(self, request):
        if type(request) == self.input_type:
            input = request.create_inference_request()
            return InferRequest(model_name=self.name, infer_inputs=[input])
        else:
            raise TypeError(f"Request type not supported, supported types are: {self.input_type}")

    def to_response(self, inference_response):
        return self.output_type(inference_response).create_response()


class WDTaggerModel(Model):
    def __init__(self):
        super().__init__(
            name="WdTaggerModel",
            input_type=request_types.ImageRequest,
            output_type=response_types.ObjectResponse,
            endpoint="localhost:8081",
        )


class TextToImageModel(Model):
    def __init__(self):
        super().__init__(
            name="TextToImageModel",
            input_type=request_types.TextToImageRequest,
            output_type=response_types.TextToImageResponse,
            endpoint="localhost:8081",
        )


class ImageToImageModel(Model):
    def __init__(self):
        super().__init__(
            name="ImageToImageModel",
            input_type=request_types.ImageRequest,
            output_type=response_types.TextToImageResponse,
            endpoint="localhost:8083",
        )

class ImageEmbeddingModel(Model):
    def __init__(self):
        super().__init__(
            name="ImageEmbeddingModel",
            input_type=request_types.ImageRequest,
            output_type=response_types.EmbeddingResponse,
            endpoint="localhost:8080",
        )

class ImageClassificationModel(Model):
    def __init__(self):
        super().__init__(
            name="ImageClassificationModel",
            input_type=request_types.ImageRequest,
            output_type=response_types.ObjectResponse,
            endpoint="localhost:8080",
        )