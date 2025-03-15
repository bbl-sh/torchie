from dataloader import load_dataloaders
from model import load_model
from trainer import trainer
from config import load_config
import mlflow
import mlflow.pytorch

mlflow.set_experiment("basics")
mlflow.pytorch.autolog()
with mlflow.start_run():
    configs = load_config("configs/config.yaml")
    dataloader = load_dataloaders(path = configs["data_path"], label = "sum", batch_size= 16)
    model = load_model()
    # for features, labels in dataloader:
    #     print(features)
    #     print(model)

    trainer(dataloader = dataloader,model= model,  NoEpoch=4)
