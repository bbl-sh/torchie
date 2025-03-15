from dataloader import load_dataloaders
from model import load_model
import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
def trainer(dataloader, model, NoEpoch = 4):

    for epoch in range(NoEpoch):
        training_loss = 0

        model.train()

        for features, labels in dataloader:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(features)
            criterion = nn.MSELoss()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        print(f'loss {training_loss}')
        mlflow.log_metric("train_loss", training_loss, step=epoch)
