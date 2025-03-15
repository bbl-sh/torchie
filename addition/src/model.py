import torch.nn as nn
class AdditionModel(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1) -> None:
        super(AdditionModel, self).__init__();
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        self.output = self.input_layer(x)
        return self.output

def load_model():
    model = AdditionModel()
    return model
