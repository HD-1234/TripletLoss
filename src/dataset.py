import random
import os
from collections import defaultdict

import pypdfium2 as pdfium
import torch

from PIL import Image
from typing import List, Tuple, Dict, Union

from torch.utils.data.dataset import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, path: str, img_size: int, max_scale_factor: int = 1, augmentation: bool = False) -> None:
        """
        Initializes the base dataset.

        Args:
            path (str): The directory containing images.
            img_size (int): The image size.
            max_scale_factor (int): The maximum scale factor for calculating the final image resolution.
            augmentation (bool): Whether to apply data augmentation or not.
        """
        super().__init__()
        self.size = img_size * max_scale_factor

        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomRotation(
                    degrees=(-2, 2)
                ),
                transforms.ColorJitter(
                    brightness=(0.85, 1.0),
                    contrast=(0.85, 1.0),
                    saturation=(0.85, 1.0),
                    hue=(-0.05, 0.05)
                ),
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(1.5, 2.0)
                ),
                transforms.RandomAdjustSharpness(
                    sharpness_factor=2,
                    p=0.5
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.1,
                    p=1.0
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist.")

        self.paths = self._get_files(path)

    @staticmethod
    def _get_files(path: str) -> dict:
        """
        Gets a dictionary of image paths grouped by template.

        Args:
            path (str): The root directory containing images.

        Returns:
            dict: A dictionary where keys are template names and values are lists of image paths.
        """
        # Allowed file extensions
        allowed_file_ext = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']

        paths = defaultdict(list)
        for root, dirs, files in os.walk(path):
            if not dirs:
                folder_name = os.path.basename(root)
                for file in files:
                    # Only add files with the allowed file extension
                    _, ext = os.path.splitext(file)
                    if ext not in allowed_file_ext:
                        continue
                    paths[folder_name].append(os.path.join(root, file))
        return paths

    @staticmethod
    def _get_templates_with_multiple_samples(paths: dict) -> List[str]:
        return [t for t, files in paths.items() if len(files) >= 2]

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """
        Loads an image from a file path.

        Args:
            path (str): The path to the file.

        Returns:
            Image.Image: A PIL Image.
        """
        # Load image file formats using PIL
        _, ext = os.path.splitext(path)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff']:
            image = Image.open(path).convert('RGB')
        # Open the PDFs using pypdfium2
        elif ext.lower() == '.pdf':
            # Open PDF
            pdf = pdfium.PdfDocument(input=path, autoclose=True)

            # Render first page and convert the page to PIL format
            image = pdf[0].render(scale=300/72).to_pil()

            # Ensure the image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Close PDF
            pdf.close()
        else:
            raise ValueError(f"Extension '{ext}' in '{path}' is not a supported file format.")

        return image

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resizes the image to the specified size.

        Args:
            image (Image.Image): The image that will be resized.

        Returns:
            Image.Image: A PIL image augmented and in the appropriate size.
        """
        # Resize Image
        image_resized = image.resize((self.size, self.size))
        return image_resized


class TripletDataset(BaseDataset):
    def __init__(self, path: str, img_size: int, max_scale_factor: int = 1, augmentation: bool = False) -> None:
        """
        Initializes the triplet dataset.

        Args:
            path (str): The directory containing images.
            img_size (int): The image size.
            max_scale_factor (int): The maximum scale factor for calculating the final image resolution.
            augmentation (bool): Whether to apply data augmentation or not.
        """
        super().__init__(path=path, img_size=img_size, max_scale_factor=max_scale_factor, augmentation=augmentation)
        self.img_size = img_size
        self.max_scale_factor = max_scale_factor

        self.templates = list(self.paths.keys())
        self.index_to_template = {i: p for i, p in enumerate([p for template in self.paths.values() for p in template])}
        self.templates_with_multiple_samples = self._get_templates_with_multiple_samples(self.paths)

    @staticmethod
    def _get_templates_with_multiple_samples(paths: dict) -> List[str]:
        return [t for t, files in paths.items() if len(files) >= 2]

    def get_paths(self, index: int) -> Tuple[Tuple[str, str, str], Tuple[int, int, int]]:
        """
        Get the file paths and labels for the anchor, positive and negative images.

        Args:
            index (int): The index of the template to use for anchor and positive samples.

        Returns:
            Tuple: A tuple containing the file paths and labels.
        """
        # Get the template name corresponding to the given index
        template = self.index_to_template[index]

        # Randomly select another template if the template has only one sample
        if template not in self.templates_with_multiple_samples:
            template = random.choice(self.templates_with_multiple_samples)

        # Randomly select an anchor and a positive sample from the chosen template
        anchor, positive = random.sample(self.paths[template], k=2)

        # Randomly select a negative sample from a different template
        template_negative = random.choice([t for t in self.templates if t != template])
        negative = random.choice(self.paths[template_negative])

        # Get the labels for the anchor, positive, and negative samples
        labels = self.templates.index(template), self.templates.index(template), self.templates.index(template_negative)

        # Pack the paths
        paths = (anchor, positive, negative)

        return paths, labels

    def __len__(self) -> int:
        return len(self.index_to_template)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Gets a triplet of images.

        Args:
            index (int): The index of the template to use for anchor and positive samples.

        Returns:
            Dict: A dict containing the tensor of the anchor, positive and negative images, the corresponding paths
            and another tensor for the corresponding labels.
        """
        # Get the paths and labels for the anchor, positive, and negative images
        paths, labels = self.get_paths(index)

        # Unpack the paths
        anchor_path, positive_path, negative_path = paths

        # Load the images
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)

        # Resize the images
        anchor = self._resize_image(anchor)
        positive = self._resize_image(positive)
        negative = self._resize_image(negative)

        # Normalize the images and transform them to a tensor. Eventually augment the images.
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return dict(
            img_size=self.img_size,
            max_scale_factor=self.max_scale_factor,
            paths=paths,
            data=torch.stack((anchor, positive, negative)),
            labels=labels_tensor
        )


class InferenceDataset(BaseDataset):
    def __init__(self, path: str, img_size: int) -> None:
        """
        Initializes the dataloader.

        Args:
            path (str): The directory containing images.
            img_size (int): The image size.
        """
        super().__init__(path=path, img_size=img_size)

        self.paths = [p for lst in self._get_files(path).values() for p in lst]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        Gets an image and its path.

        Args:
            index (int): The index of the image.

        Returns:
            Tuple: A tuple containing the image tensor and the file path.
        """
        # Load the image
        image = self._load_image(self.paths[index])

        # Resize the image
        image = self._resize_image(image)

        # Normalize the image and transform it to a tensor.
        image = self.transform(image)

        return image, self.paths[index]
