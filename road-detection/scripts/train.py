import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.data_utils import YoloDataset, collate_fn

# Configurations
train_images = "data/images/train/"
train_labels = "data/labels/train/"
val_images = "data/images/val/"
val_labels = "data/labels/val/"
batch_size = 8
num_epochs = 10
learning_rate = 0.001
checkpoint_dir = "./models/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for saving models
os.makedirs(checkpoint_dir, exist_ok=True)

# Load Dataset and DataLoader
train_dataset = YoloDataset(train_images, train_labels, transform=None)
val_dataset = YoloDataset(val_images, val_labels, transform=None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define Pretrained Model
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    return model

num_classes = 11  # 10 classes + background
model = get_model(num_classes)
model = model.to(device)

# Optimizer and Scheduler
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training Function
def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [
            {
                "boxes": torch.tensor(
                    [[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for _, x, y, w, h in labels]
                ).to(device),
                "labels": torch.tensor([int(cls) for cls, _, _, _, _ in labels]).to(device),
            }
            for labels in targets
        ]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(train_loader)

# Validation Function
@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [
            {
                "boxes": torch.tensor(
                    [[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for _, x, y, w, h in labels]
                ).to(device),
                "labels": torch.tensor([int(cls) for cls, _, _, _, _ in labels]).to(device),
            }
            for labels in targets
        ]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / len(val_loader)

# Main Training Loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training phase
    train_loss = train_one_epoch(model, train_loader, optimizer)
    print(f"Train Loss: {train_loss}")

    # Validation phase
    val_loss = validate(model, val_loader)
    print(f"Validation Loss: {val_loss}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch + 1}.pth")
    print(f"Model checkpoint saved at epoch {epoch + 1}")

    # Update scheduler
    scheduler.step()

print("Training completed.")
