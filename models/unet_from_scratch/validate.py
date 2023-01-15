import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from models.unet_from_scratch.UNET_MODEL_FROM_SCRATCH import UNET_FROM_SCRATCH
from models.unet_from_scratch.utils import get_loaders, save_checkpoint, print_measurements, save_predictions_as_imgs, \
    load_checkpoint

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_IMG_DIR = 'E:\CVDL\MyApp\models\data\\images'
TRAIN_MASK_DIR = 'E:\CVDL\MyApp\models\data\labels'
VAL_IMG_DIR = 'E:\CVDL\MyApp\models\data\\validationimages'
VAL_MASK_DIR = 'E:\CVDL\MyApp\models\data\\validationlabels'

def validate():
    train_transfrom = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0
                        ),
            ToTensorV2()
        ]
    )

    val_transfrom = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0
                        ),
            ToTensorV2()
        ]
    )
    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR,
                                           TRAIN_MASK_DIR,
                                           VAL_IMG_DIR,
                                           VAL_MASK_DIR,
                                           BATCH_SIZE,
                                           train_transfrom,
                                           val_transfrom,
                                           NUM_WORKERS,
                                           PIN_MEMORY
                                           )
    model = UNET_FROM_SCRATCH(in_channels=3, out_channels=1).to(DEVICE)
    if LOAD_MODEL:
        load_checkpoint(torch.load('check_unet_scratch.pth.tar'), model)
    print_measurements(val_loader, model)
    save_predictions_as_imgs(val_loader, model, folder="validate_unet/", device=DEVICE)




if __name__=='__main__':
    print("Starting validate")
    validate()
    print("End validate")
