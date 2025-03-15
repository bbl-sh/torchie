import torch
import torchvision.transforms as T
from utils.data_utils import load_data

train_images = "data/train/images/"
train_labels = "data/train/labels/"
val_images = "data/val/images/"
val_labels = "data/val/labels"

batch_size = 16

transform = T.Compose([
        T.ToPILImage(),
        T.Resize((640,640)),        # Resize image to the desired size (e.g., 640x640)
        T.ToTensor(),                # Convert image to PyTorch tensor (C, H, W)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

train_loader = load_data(train_images, train_labels, transform = transform, batch_size=batch_size, shuffle=True)
val_loader = load_data(val_images, val_labels, batch_size=batch_size, shuffle=False)


## For validating if the dataloader has properly loaded the files
# for images, labels in train_loader:
#     print(f"Batch images shape: {images.shape}")
#     print(f"Batch labels: {labels}")
#     break
#
