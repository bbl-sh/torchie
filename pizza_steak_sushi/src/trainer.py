import torch
import torch.nn as nn
import torch.optim as optim
def trainer(dataloader, model, NoEpoch = 5):
    for epoch in range(NoEpoch):
        running_loss = 0
        correct = 0
        preds = 0
        total = 0
        for images, labels in dataloader:
            output = model(images)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(running_loss)
