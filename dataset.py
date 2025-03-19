import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultiModalDataset(Dataset):
    def __init__(self, root_dir, modalities, transform=None):
        self.root_dir = root_dir
        self.modalities = modalities
        self.transform = transform

        self.image_names = sorted(os.listdir(os.path.join(root_dir, modalities[0])))

    def __len__(self):
        return len(self.image_names)*2

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        images = {}

        for modality in self.modalities:
            img_path = os.path.join(self.root_dir, modality, img_name)
            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img)

            images[modality] = img

        return images


def get_data_loaders(data_dir, modalities, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_transform = A.Compose(
        [
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=1.0),
            A.OneOf([A.Rotate(limit=30, p=0.7), A.Rotate(limit=45, p=0.3)], p=0.7),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=0.5)], p=1.0),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )

    dataset = MultiModalDataset(data_dir, modalities, transform=test_transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
