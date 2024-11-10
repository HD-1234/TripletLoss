import random
import os
import torch
import pikepdf

from PIL import Image
from typing import List, Tuple

from pikepdf import PdfImage
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class TripletDataset(Dataset):
    def __init__(self, path: str, size: int, augmentation: bool = False) -> None:
        """
        Initializes the dataloader.

        Args:
            path (str): The directory containing images.
            size (int): The image size.
            augmentation (bool): Whether to apply data augmentation or not.
        """
        super().__init__()

        self.size = size

        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=(-2, 2)),
                transforms.ColorJitter(brightness=(0.85, 1.0), contrast=(0.85, 1.0), saturation=(0.85, 1.0),
                                       hue=(-0.05, 0.05)),
                transforms.GaussianBlur(kernel_size=3, sigma=(1.5, 2.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist.")

        self.paths = self._get_files(path)
        self.templates = list(self.paths.keys())
        self.index_to_template = {i: p for i, p in enumerate([p for template in self.paths.values() for p in template])}
        self.templates_with_multiple_samples = self._get_templates_with_multiple_samples(self.paths)

    @staticmethod
    def _get_files(path: str) -> dict:
        """
        Gets a dictionary of image paths grouped by template.

        Args:
            path (str): The root directory containing images.

        Returns:
            dict: A dictionary where keys are template names and values are lists of image paths.
        """
        paths = dict()
        for root, dirs, files in os.walk(path):
            if not dirs:
                folder_name = os.path.basename(root)
                if folder_name not in paths:
                    paths[folder_name] = list()
                paths[folder_name].extend([os.path.join(root, file) for file in files])
        return paths

    @staticmethod
    def _get_templates_with_multiple_samples(paths: dict) -> List[str]:
        return [t for t, files in paths.items() if len(files) >= 2]

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """
        Gets a file path.

        Args:
            path (str): The path to the image.

        Returns:
            Image.Image: A PIL Image.
        """
        # Load image file formats using PIL
        _, ext = os.path.splitext(path)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff']:
            image = Image.open(path).convert('RGB')
        # Open the PDFs using pikepdf
        elif ext.lower() == '.pdf':
            with pikepdf.Pdf.open(path) as pdf:
                # Get the first page
                page = pdf.pages[0]

                # Get the page key
                page_key = list(page.images.keys())[0]

                # Render the page into a PIL Image
                raw_page = page.images[page_key]
                pdf_image = PdfImage(raw_page)
                image = pdf_image.as_pil_image()

                # Ensure the image is in RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
        else:
            raise ValueError(f"Extension '{ext}' in '{path}' is not a supported file format.")

        return image

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Gets a PIL Image.

        Args:
            image (Image.Image): The image that will be resized.

        Returns:
            Image.Image: A PIL image augmented and in the appropriate size.
        """
        # Resize Image
        image_resized = image.resize((self.size, self.size))

        return image_resized

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets a triplet of images.

        Args:
            index (int): The index of the template to use for anchor and positive samples.

        Returns:
            Tuple: A Tuple containing a tensor of the anchor, positive and negative images and another tensor for the
            corresponding labels.
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
        anchor = self.resize_image(anchor)
        positive = self.resize_image(positive)
        negative = self.resize_image(negative)

        # Normalize the images and transform them to a tensor. Eventually augment the images.
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return torch.stack((anchor, positive, negative)), labels_tensor
