import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader


class RoadDetectionDataset(Dataset):
    def __init__(self, image_dir, labels_dir, transform = None):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(self.image_dir))
        self.labels_files = sorted(os.listdir(self.labels_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.labels_dir, self.labels_files[idx])

        with open(label_path, 'r') as f:
            labels = []
            for line in f.readlines():
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append([cls, x_center, y_center, width, height])

            labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        # To remove the warning convert to numpy  img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), labels

def collate_fn(batch):
    """
    Custom collate function for handling batches with variable-length labels.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets


def load_data(image_dir, labels_dir, transform = None, shuffle = True, num_workers = 0, batch_size = 1):
    dataset = RoadDetectionDataset(image_dir, labels_dir, transform)

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return dataloader
