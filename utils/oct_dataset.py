import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class OCTDataset(Dataset):
    def __init__(
        self, root_dir, indices=None, train=True, is_gentuity=False, transform=None, for_augmentation=False
    ):

        self.root_dir = Path(root_dir)
        self.train = train
        self.is_gentuity = is_gentuity
        self.transform = transform

        if self.is_gentuity:
            # Gentuity dataset has separate train and test folders
            split_dir = "train" if self.train else "test"
            self.images_dir = self.root_dir / split_dir / "images"
            self.masks_dir = self.root_dir / split_dir / "annotations"
            self.samples = sorted(self.images_dir.glob("*.tiff"))
        else:
            # Terumo dataset has only train data
            self.images_dir = self.root_dir / "train" / "images"
            self.masks_dir = self.root_dir / "train" / "annotations"
            self.samples = sorted(self.images_dir.glob("*.tiff"))

        # Filter image paths using indices if provided
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        else:
            self.samples = sorted(self.images_dir.glob("*.tiff"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Keep it as a PIL Image
        # Get image height and width
        width, height = image.size

        # Load the corresponding mask
        mask_path = self.masks_dir / f"{image_path.stem}.json"
        with open(mask_path, "r") as f:
            mask_data = json.load(f)

        # Create a binary mask (0 and 1 values)
        mask = np.zeros(
            (height, width), dtype=np.uint8
        )  # image.size gives (width, height)
        for coord in mask_data["mask"]:
            x, y = coord
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]:
                mask[x, y] = 1

        mask = np.clip(mask, 0, 1).astype(np.uint8)

        mask = Image.fromarray(mask * 255)

        # Apply the transformation if available
        if self.transform:
            if self.for_augmentation == False:
                # Convert image and mask to Tensor
                image = self.transform(image)
                mask = self.transform(mask)
            else:
                ## For data augmentation ##
                image, mask = self.transform(image, mask)
        
        mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask  # Add channel if needed
        # Add channel dimension for mask
        if self.for_augmentation == False:
            unique_id = mask_data["unique_id"]
        else:
            unique_id = 4

        return image, mask, image_path.stem, unique_id  # Returning image filename too
