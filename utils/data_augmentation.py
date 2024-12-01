from torchvision.transforms import v2

class SynchronizedRandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, label):
        # Generate a random angle for rotation
        angle = v2.RandomRotation.get_params([-self.degrees, self.degrees])

        # Rotate both image and label by the same angle
        img = v2.functional.rotate(img, angle)
        label = v2.functional.rotate(label, angle)

        return img, label


class DataAugmentTransform:
    def __init__(self, tensor_transform, color_transform, random_rotation):
        # Augmentation transform composition
        tensor_transform = v2.Compose([
            v2.Resize((256, 256), interpolation=Image.NEAREST),
            v2.ToTensor(),
        ])

        color_transform = v2.Compose([
            v2.ColorJitter(contrast=0.5, brightness=0.5, saturation=None, hue=None)
        ])

        random_rotation = SynchronizedRandomRotation(degrees=180)

        self.tensor_transform = tensor_transform
        self.color_transform = color_transform
        self.random_rotation = random_rotation

    def __call__(self, img, label):
        img, label = self.tensor_transform(img, label)
        img = self.color_transform(img)
        img, label = self.random_rotation(img, label)

        return img, label