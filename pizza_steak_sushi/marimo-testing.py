import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from PIL import Image
    import os
    return Image, os, torch


@app.cell
def _(os):
    names = sorted(os.listdir("./data/train"))
    return (names,)


@app.cell
def _(names):
    names
    return


@app.cell
def _(names):
    class_to_idx = {cls_name: cls_name for _, cls_name in enumerate(names)}
    return (class_to_idx,)


@app.cell
def _(class_to_idx):
    class_to_idx
    return


@app.cell
def _():
    x = []
    return (x,)


@app.cell
def _(class_to_idx, names, os, x):
    for cls_name in names:
        cls_dir = os.path.join("./data/train", cls_name)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            x.append((img_path, class_to_idx[cls_name]))
    return cls_dir, cls_name, img_name, img_path


@app.cell
def _(x):
    image, path = x[0]
    return image, path


@app.cell
def _(image):
    image
    return


@app.cell
def _(path):
    path
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
