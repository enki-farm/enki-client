from abc import abstractmethod

import orjson

import io

from PIL import Image

from kserve import InferResponse

class Response():
    """
    Base class for responses from inference.
    """
    def __init__(self, inference_response):
        """
        Initialize a Response instance.

        Args:
            inference_response (InferResponse): The inference response from KServe.
        """
        self.inference_response = InferResponse.from_grpc(inference_response)

    @abstractmethod
    def create_response(self):
        """
        Create a response from the inference response.

        This method should be overridden by subclasses.

        Returns:
            The created response.
        """
        pass


class ObjectResponse(Response):
    """
    Class for object responses from inference.
    """
    def create_response(self):
        """
        Create an object response from the inference response.

        Returns:
            dict: The created object response.
        """
        obj = orjson.loads(self.inference_response.outputs[0].data[0])
        return obj
    

class ImageResponse(Response):
    """
    Class for image responses from inference.
    """
    def create_response(self):
        """
        Create an image response from the inference response.

        Returns:
            PIL.Image.Image: The created image response.
        """
        image_bytes = self.inference_response.outputs[0].data[0]
        return Image.open(io.BytesIO(image_bytes))
    

class TextToImageResponse(Response):
    """
    Class for text-to-image responses from inference.
    """
    def create_response(self):
        """
        Create a text-to-image response from the inference response.

        Returns:
            dict: The created text-to-image response. The dictionary has the following structure:
                {
                    "image": PIL.Image.Image,  # The generated image.
                    "metadata": {
                        "seed": int,  # The seed used for generating the image.
                    }
                }
        """
        image_bytes = self.inference_response.outputs[0].data[0]
        return {
            "image": Image.open(io.BytesIO(image_bytes)),
            "metadata": orjson.loads(self.inference_response.outputs[0].data[1])
        }
    
class EmbeddingResponse(Response):
    """
    Class for embedding responses from inference.
    """
    def create_response(self):
        """
        Create an embedding response from the inference response.

        Returns:
            numpy.ndarray: The created embedding response.
        """
        embedding = self.inference_response.outputs[0].data[0]
        return embedding