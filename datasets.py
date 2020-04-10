import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", face_part="eyes"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "images/non-makeup") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "images/makeup") + "/*.*"))

        self.files_A_mask = sorted(glob.glob(os.path.join(root, "%s/non-makup" % face_part) + "/*.*"))
        self.files_B_mask = sorted(glob.glob(os.path.join(root, "%s/makeup" % face_part) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_A_mask = Image.open(self.files_A_mask[index % len(self.files_A_mask)])

        if self.unaligned:
            idx = random.randint(0, len(self.files_B) - 1)
            image_B = Image.open(self.files_B[idx])
            image_B_mask = Image.open(self.files_B_mask[idx])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
            image_B_mask = Image.open(self.files_B_mask[index % len(self.files_B_mask)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_A_mask = self.transform(image_A_mask)
        item_B_mask = self.transform(image_B_mask)
        return {"A": item_A, "B": item_B, "A_mask": item_A_mask, "B_mask": item_B_mask}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))