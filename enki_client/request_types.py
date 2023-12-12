from abc import ABC, abstractmethod

import orjson

from kserve import InferInput

class Request(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_inference_request(self):
        pass


class ImageRequest(Request):
    def __init__(self, image):
        """
        Initialize an ImageRequest instance.

        Args:
            image (str): The path to the image file.
        """
        self.image = image

    def create_inference_request(self):
        """
        Create an input for inference from the image file.

        Returns:
            InferInput: The input for inference.
        """
        # pass url directly -> inference service will download
        if self.image.startswith("http"):
            data = self.image
        else:
            with open(self.image, "rb") as f:
                data = f.read()

        input = InferInput(name="image", shape=[1], datatype="BYTES", data=[data])
        return input
    

class TextRequest(Request):
    def __init__(self, text):
        """
        Initialize a TextRequest instance.

        Args:
            text (str): The text for inference.
        """
        self.text = text

    def create_inference_request(self):
        """
        Create an input for inference from the text.

        Returns:
            InferInput: The input for inference.
        """
        input = InferInput(name="text", shape=[1], datatype="BYTES", data=[self.text])
        return input

class TextToImageRequest(Request):
    def __init__(self,
                 prompt: str,
                 image: str = None,
                 image_strength: float = 0.5,
                 neg_prompt:str = "",
                 num_inference_steps:int = 25,
                 guidance_scale:float = 7.5,
                 width:int = 1024,
                 height:int = 1024, 
                 seed:int = 0,
        ):
        """
        Initialize a TextToImageRequest instance.

        Args:
            prompt (str): The prompt for the image generation.
            image (str): The path to the image file.
            image_strength (float, optional): The strength of the image. Defaults to 0.5.
            neg_prompt (str, optional): The negative prompt for the image generation. Defaults to "(worst quality, low quality:1.4)".
            num_inference_steps (int, optional): The number of inference steps. Defaults to 25.
            guidance_scale (float, optional): The guidance scale. Defaults to 7.5.
            width (int, optional): The width of the generated image. Defaults to 512.
            height (int, optional): The height of the generated image. Defaults to 512.
            seed (int, optional): The seed for the random number generator. Defaults to 0.
        """
        self.prompt = prompt
        self.image = image
        self.image_strength = image_strength
        self.neg_prompt = neg_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.seed = seed

    def create_inference_request(self):
        """
        Create an input for inference from the text and image parameters.

        Returns:
            InferInput: The input for inference.
        """
        data = { 
            "prompt": self.prompt,
            "neg_prompt": self.neg_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "seed": self.seed
        }
        if self.image:
            with open(self.image, "rb") as f:
                image_data = f.read()
            data["image_data"] = image_data
            data["image_strength"] = self.image_strength

        data_bytes = orjson.dumps(data)
        input = InferInput(name="prompt", shape=[1], datatype="BYTES", data=[data_bytes])
        return input
