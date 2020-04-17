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
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "images/non-makeup") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "images/makeup") + "/*.*"))

        self.files_A_mask_eyes = sorted(glob.glob(os.path.join(root, "%s/non-makup" % "eyes") + "/*.*"))
        self.files_A_mask_lips = sorted(glob.glob(os.path.join(root, "%s/non-makup" % "lips") + "/*.*"))
        self.files_A_mask_face = sorted(glob.glob(os.path.join(root, "%s/non-makup" % "face") + "/*.*"))
        self.files_A_mask_bg = sorted(glob.glob(os.path.join(root, "%s/non-makup" % "background") + "/*.*"))

        self.files_B_mask_eyes = sorted(glob.glob(os.path.join(root, "%s/makeup" % "eyes") + "/*.*"))
        self.files_B_mask_lips = sorted(glob.glob(os.path.join(root, "%s/makeup" % "lips") + "/*.*"))
        self.files_B_mask_face = sorted(glob.glob(os.path.join(root, "%s/makeup" % "face") + "/*.*"))

    def __getitem__(self, index):
        idx_A = index % len(self.files_A)
        image_A = Image.open(self.files_A[idx_A])
        image_A_mask_eyes = Image.open(self.files_A_mask_eyes[idx_A])
        image_A_mask_lips = Image.open(self.files_A_mask_lips[idx_A])
        image_A_mask_face = Image.open(self.files_A_mask_face[idx_A])
        image_A_mask_bg = Image.open(self.files_A_mask_bg[idx_A])

        # File B - Eyes
        idx = random.randint(0, len(self.files_B) - 1)
        image_B = Image.open(self.files_B[idx])
        image_B_mask = Image.open(self.files_B_mask_eyes[idx])

        # File C - Lips
        idx = random.randint(0, len(self.files_B) - 1)
        image_C = Image.open(self.files_B[idx])
        image_C_mask = Image.open(self.files_B_mask_lips[idx])

        # File D - Face
        idx = random.randint(0, len(self.files_B) - 1)
        image_D = Image.open(self.files_B[idx])
        image_D_mask = Image.open(self.files_B_mask_face[idx])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        if image_C.mode != "RGB":
            image_C = to_rgb(image_C)
        if image_D.mode != "RGB":
            image_D = to_rgb(image_D)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_C = self.transform(image_C)
        item_D = self.transform(image_D)

        item_A_mask_eyes = self.transform(image_A_mask_eyes)
        item_A_mask_lips = self.transform(image_A_mask_lips)
        item_A_mask_face = self.transform(image_A_mask_face)
        item_A_mask_bg = self.transform(image_A_mask_bg)

        item_B_mask = self.transform(image_B_mask)
        item_C_mask = self.transform(image_C_mask)
        item_D_mask = self.transform(image_D_mask)
        return {"A": item_A, "B": item_B, "C": item_C, "D": item_D, "A_mask_bg": item_A_mask_bg, \
            "A_mask_eyes": item_A_mask_eyes, "A_mask_lips": item_A_mask_lips, "A_mask_face": item_A_mask_face, \
            "B_mask": item_B_mask, "C_mask": item_C_mask, "D_mask": item_D_mask}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))