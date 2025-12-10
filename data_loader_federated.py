"""Federated Data Loader for Multimodal Learning with Missing Modalities."""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple

LABEL_COLUMNS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]


class MultimodalFederatedDataset(Dataset):
    """Dataset for federated multimodal learning."""

    def __init__(self, csv_path: str, base_path: str = "Fed_Data",
                 is_multimodal: bool = True, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_path = base_path
        self.is_multimodal = is_multimodal
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.base_path, "images", row['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)

        text = ""
        has_text = False
        if self.is_multimodal and pd.notna(row['report_path']) and row['report_path']:
            report_path = os.path.join(self.base_path, "reports", row['report_path'])
            try:
                with open(report_path, 'r') as f:
                    text = f.read().strip()
                has_text = True
            except Exception as e:
                print(f"Error loading report {report_path}: {e}")

        labels = torch.tensor(
            [float(row[col]) if pd.notna(row[col]) else 0.0 for col in LABEL_COLUMNS],
            dtype=torch.float32
        )

        return {
            'image': image,
            'text': text,
            'labels': labels,
            'has_text': has_text,
            'has_image': True,
            'patient_id': row['patient_id']
        }


class PadChestTestDataset(Dataset):
    """PadChest test dataset."""

    def __init__(self, csv_path: str = "PadChest_Data/padchest_test.csv", transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)

        text = row['report'] if pd.notna(row['report']) else ""

        labels = torch.tensor(
            [float(row[col]) if pd.notna(row[col]) else 0.0 for col in LABEL_COLUMNS],
            dtype=torch.float32
        )

        return {
            'image': image,
            'text': text,
            'labels': labels,
            'has_text': bool(text),
            'has_image': True,
            'image_id': row['image_id']
        }


def get_transforms(is_train=True):
    """Get image transforms with conservative augmentation for medical imaging."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_federated_dataloaders(
    num_multimodal_clients: int = 2,
    num_unimodal_clients: int = 8,
    batch_size: int = 32,
    base_path: str = "Fed_Data"
) -> Tuple[List[DataLoader], List[bool]]:
    """Create federated dataloaders. Clients 0-7: unimodal, Clients 8-9: multimodal."""
    total_clients = num_multimodal_clients + num_unimodal_clients
    client_loaders = []
    is_multimodal_list = []
    transform = get_transforms(is_train=True)

    for client_id in range(total_clients):
        is_multimodal = client_id >= num_unimodal_clients
        csv_path = os.path.join(base_path, "splits", f"client_{client_id}.csv")

        dataset = MultimodalFederatedDataset(
            csv_path=csv_path,
            base_path=base_path,
            is_multimodal=is_multimodal,
            transform=transform
        )

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )

        client_loaders.append(loader)
        is_multimodal_list.append(is_multimodal)
        print(f"Client {client_id}: {'Multimodal' if is_multimodal else 'Unimodal'}, Samples: {len(dataset)}")

    return client_loaders, is_multimodal_list


def create_test_dataloader(batch_size: int = 32) -> DataLoader:
    """Create test dataloader from PadChest."""
    transform = get_transforms(is_train=False)
    dataset = PadChestTestDataset(csv_path="PadChest_Data/padchest_test.csv", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test set: {len(dataset)} samples")
    return loader


if __name__ == "__main__":
    print("Testing data loading...")
    client_loaders, is_multimodal = create_federated_dataloaders()
    test_loader = create_test_dataloader()
    print(f"\nTotal clients: {len(client_loaders)}")
    print(f"Multimodal clients: {sum(is_multimodal)}")
    print(f"Unimodal clients: {len(is_multimodal) - sum(is_multimodal)}")
