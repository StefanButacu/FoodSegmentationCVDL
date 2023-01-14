import pathlib
import numpy as np
import torch
import torchvision.utils
from torch.utils.data import DataLoader

from models.unet_from_scratch.dataset import FoodDataset
from models.unet_from_scratch.visual import DatasetViewer


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("=>saving checkpoint to",  filename)
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames



def get_loaders(TRAIN_IMG_DIR,
                TRAIN_MASK_DIR,
                VAL_IMG_DIR,
                VAL_MASK_DIR,
                BATCH_SIZE,
                train_transfrom,
                val_transfrom,
                NUM_WORKERS,
                PIN_MEMORY
                ):
    train_ds = FoodDataset(inputs=get_filenames_of_path(pathlib.Path(TRAIN_IMG_DIR)),
                           targets=get_filenames_of_path(pathlib.Path(TRAIN_MASK_DIR)),
                           transform=train_transfrom)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=False)
    VAL_IMG_DIR = TRAIN_IMG_DIR
    VAL_MASK_DIR = TRAIN_MASK_DIR
    val_ds = FoodDataset(inputs=get_filenames_of_path(pathlib.Path(VAL_IMG_DIR)),
                       targets=get_filenames_of_path(pathlib.Path(VAL_MASK_DIR)),
                       transform=val_transfrom)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=False)

    return train_loader, val_loader


def print_measurements(loader, model, device="cpu"):
    num_correct = 0
    num_pixel = 0
    dice_score = 0
    ioc = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)   # B H W -> B 1 H W
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()   # 0.5 -> threshold
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            batch_ioc = calculate_ioc1(preds, y)
            ioc += batch_ioc

    print(f"Got {num_correct}/{num_pixel} with acc {num_correct/num_pixel * 100:.2f}", )
    print(f"Got dice_score:{dice_score/len(loader)} ")
    print(f"Got ioc1:{ioc} ")


def calculate_ioc1(preds, targets):
    ioc = 0
    for i in range(len(preds)):
        pred = preds[i][0].numpy()
        target = targets[i].numpy()
        overlap = pred * target # logical and
        union = pred + target  # logical or
        ioc += overlap.sum() / float(union.sum())
    return ioc / len(preds)


def save_predictions_as_imgs(loader, model, folder, device="cpu"):
    print("Saving images to....", folder)
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}/prediction_{idx}.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/truth_{idx}.png")
            torchvision.utils.save_image(x, f"{folder}/x_{idx}.png")
    print("Finish saving images...")
    model.train()
