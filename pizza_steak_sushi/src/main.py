from dataloader import load_dataloader
from trainer import trainer
from model import load_model

dataloader = load_dataloader()
model = load_model()
trainer(dataloader = dataloader, model = model)
