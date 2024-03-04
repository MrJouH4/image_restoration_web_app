import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

class Div2kBlurDataset(Dataset):
    def __init__(self, div2k_path, blur_levels, patch_size=None, train=True, train_ratio=0.75):
        self.image_list = []
        self.blur_levels = blur_levels
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        for i in os.listdir(div2k_path):
            self.image_list.append(os.path.join(div2k_path, i))

        total_samples = len(self.image_list)
        train_size = int(train_ratio * total_samples)
        val_size = total_samples - train_size

        if train:
            self.image_list = self.image_list[:train_size]
        else:
            self.image_list = self.image_list[-val_size:]

    def __len__(self):
        return len(self.image_list)

    def extract_patches(self, image):
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        patches = unfold(image)
        num_patches = patches.shape[-1]
        patches = patches.view(3, self.patch_size, self.patch_size, num_patches)
        patches = patches.permute(3, 0, 1, 2)
        return patches

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        hr_image = Image.open(img_path).convert('RGB')
        hr_image = self.transform(hr_image)

        blur_radius = np.random.choice(self.blur_levels)

        # Apply Gaussian Blur
        blur_transform = transforms.GaussianBlur(blur_radius)
        blurred_image = blur_transform(hr_image)

        if self.patch_size is not None:
            hr_image = self.extract_patches(hr_image)
            blurred_image = self.extract_patches(blurred_image)

        return blurred_image, hr_image
