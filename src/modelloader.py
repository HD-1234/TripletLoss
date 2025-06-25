from torch import nn
from src.models.resnet import *
from src.models.vit import *
from src.models.vitar import *


class ModelLoader:
    def __init__(self, model_name: str, image_size: int = 224) -> None:
        """
        Initializes the model loader.

        Args:
            model_name (str): The name of the model.
            image_size (int): The size of the input image.
        """
        self.model_name = model_name.lower()
        self.image_size = image_size

    def load_model(self) -> nn.Module:
        """
        Loads the corresponding model.

        Returns:
            nn.Module: The actual model.
        """
        model_mapping = {
            'resnet50': ResNet50,
            'resnet101': ResNet101,
            'resnet152': ResNet152,
            'resnext50': ResNeXt50,
            'resnext101': ResNeXt101,
            'vit_b_16': VisionTransformerB16,
            'vit_b_32': VisionTransformerB32,
            'vit_l_16': VisionTransformerL16,
            'vit_l_32': VisionTransformerL32,
            'vitar_b_16': VisionTransformerAnyResolutionB16,
            'vitar_l_16': VisionTransformerAnyResolutionL16
        }

        if self.model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {self.model_name}")

        # Model name to actual model
        model_type = model_mapping[self.model_name]

        # Initialize the model
        model = model_type(image_size=self.image_size)

        return model
