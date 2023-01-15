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
    iou = 0
    print("printing measure")
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
            iou += (preds * y).sum() / ((preds + y).sum() + 1e-8)

    model.train()
    print(f"Got {num_correct}/{num_pixel} with acc {num_correct/num_pixel * 100:.2f}", )
    print(f"Got dice_score:{dice_score/len(loader)} ")
    print(f"Got ioc1:{iou / len(loader)} ")


def calculate_ioc1(preds, targets):

    iou = 0
    for i in range(len(preds)):
        pred = preds[i][0].numpy()
        target = targets[i].numpy()
        overlap = pred * target # logical and
        union = pred + target  # logical or
        iou += overlap.sum() / float(union.sum())
    return iou

def iou_pytorch(preds, targets):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    preds = preds.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    targets = targets.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (preds * targets).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (preds + targets).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

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
