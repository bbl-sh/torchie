import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv('add.csv')

X = data[['x', 'y']].values
y = data['sum'].values

def train_test_split(X, y, test_size=0.2):
    split_idx = int(len(X) - test_size * len(X))

    X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
    X_test = torch.tensor(X[split_idx:], dtype=torch.float32)

    y_train = torch.tensor(y[:split_idx], dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y[split_idx:], dtype=torch.float32).view(-1, 1)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X, y)

class Addition_model(nn.Module):
    def __init__(self):
        super(Addition_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2,40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

model = Addition_model()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5000

for epoch in range(epochs):
    model.train()

    preditions = model(X_train)
    loss = criterion(preditions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if((epoch+1) % 50 == 0):
        print(f"Epoch : {epoch+1}, loss : {loss.item():.4f}")



model.eval()
with torch.no_grad():
    for epoch in range(len(X_test)):
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f"Epoch : {epoch+1}, loss : {test_loss.item():.4f}")
