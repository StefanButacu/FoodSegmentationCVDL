import numpy as np
from torch.utils import data
from PIL import Image


class FoodDataset(data.Dataset):
    def __init__(self, inputs, targets, transform=None):
        self.inputs = inputs[:100]
        self.targets = targets[:100]
        self.transform = transform
        self.counter = 0

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        image, mask = Image.open(input_ID), Image.open(target_ID)

        new_mask = [
            [1.0 if mask.getpixel((x, y))[-1] == 1 else 0.0
             for x in range(mask.width)] for y in range(mask.height)
        ]
        mask = new_mask
        image, mask = np.array(image), np.array(mask, dtype=np.float32)
        if self.transform is not None:
            # try:
                augmentations = self.transform(image=image, mask= mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            # except  ValueError:
            #     print("InputId_", input_ID)
            #     print("TargetId_", target_ID)
            #     self.counter +=1
            #     print(f"value error for the {self.counter} time")

        return image, mask

