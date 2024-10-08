from torch import nn
from src.models.resnext50 import ResNeXt50


class ModelLoader:
    def __init__(self, model_name: str, pretrained_weights: str = None) -> None:
        """
        Initializes the model loader.

        Args:
            model_name (str): The name of the model.
            pretrained_weights (int): The pre-trained weights of the model.
        """
        self.model_name = model_name.lower()
        self.pretrained_weights = pretrained_weights

    def load_model(self) -> nn.Module:
        """
        Loads the corresponding model.

        Returns:
            nn.Module: The actual model.
        """
        model_mapping = {
            'resnext50': ResNeXt50
        }

        if self.model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {self.model_name}")

        # Model name to actual model
        model_type = model_mapping[self.model_name]

        # Initialize the model
        model = model_type(pretrained_weights=self.pretrained_weights)

        return model
