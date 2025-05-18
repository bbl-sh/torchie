import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    return DataLoader, Dataset, pd, torch


@app.cell
def _(pd):
    df = pd.read_csv("data/add.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    len(df)
    return


@app.cell
def _(df):
    x = df.drop("sum",  axis = "columns").values
    #x = df.drop(columns = ["sum", "x"],  axis = "columns").values
    y = df["sum"].values
    return x, y


@app.cell
def _(x):
    x[4]
    return


@app.cell
def _(y):
    y[4]
    return


@app.cell
def _(Dataset, pd, torch):
    class AdditionDataset(Dataset):
        def __init__(self, path, labels):
            self.data = pd.read_csv(path);
            self.x = self.data.drop(labels, axis = "columns").values
            self.y = self.data[labels].values

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx): 
            x = torch.tensor(self.x[idx], dtype=torch.float32)
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y

    return (AdditionDataset,)


@app.cell
def _(AdditionDataset, DataLoader):
    dataset = AdditionDataset(path = "data/add.csv", labels  = "sum")
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)
    for features, labels in dataloader: 
        print(features)
    return dataloader, dataset, features, labels


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
