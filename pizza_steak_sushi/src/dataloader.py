from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms
class CustomDataset(Dataset):
    def __init__(self, path, annotations = ['pizza','steak', 'sushi']):
        self.path = path
        self.annotation = annotations
        self.classes = sorted(os.listdir(path))
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls in self.classes:
            cls_imgs = os.path.join(path, cls)
            for cls_img in os.listdir(cls_imgs):
                img_path = os.path.join(cls_imgs, cls_img)
                self.samples.append((img_path, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_pth, label = self.samples[idx]
        img_tensor = Image.open(img_pth).convert('RGB')
        img_tensor = self.transform(img_tensor)
        label = self.cls_to_idx[label]
        return img_tensor, label

def load_dataloader(path = "../data/train",batch_size = 1, shuffle = True):
    dataset = CustomDataset(path=path)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)
    # for img, label in dataloader:
    #     print(label)
    #     break
    return dataloader

#load_dataloader()
