import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

class GoProDataset(Dataset):
    def __init__(self, root_dir, patch_size, train=True, train_ratio=0.75):
        super().__init__()

        self.root_dir = root_dir
        self.patch_size = patch_size

        self.blurry_image_paths = []
        self.sharp_image_paths = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            for type_folder in os.listdir(folder_path):
                if type_folder == "sharp":
                    for img in os.listdir(os.path.join(root_dir, folder, type_folder)):
                        self.blurry_image_paths.append(os.path.join(root_dir, folder, "blur", img))
                        self.sharp_image_paths.append(os.path.join(root_dir, folder, "sharp", img))

        total_samples = len(self.blurry_image_paths)
        train_size = int(train_ratio * total_samples)
        val_size = total_samples - train_size

        if train:
            self.blurry_image_paths, self.sharp_image_paths = self.blurry_image_paths[:train_size], self.sharp_image_paths[:train_size]
        else:
            self.blurry_image_paths, self.sharp_image_paths = self.blurry_image_paths[-val_size:], self.sharp_image_paths[-val_size:]

    def __len__(self):
        return len(self.blurry_image_paths)

    def __getitem__(self, idx):
        blurry_image_path = self.blurry_image_paths[idx]
        sharp_image_path = self.sharp_image_paths[idx]

        blurry_image = Image.open(blurry_image_path)
        sharp_image = Image.open(sharp_image_path)

        blurry_image = transforms.ToTensor()(blurry_image)
        sharp_image = transforms.ToTensor()(sharp_image)

        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        blurry_patches = unfold(blurry_image)
        sharp_patches = unfold(sharp_image)

        num_patches = blurry_patches.shape[-1]
        blurry_patches = blurry_patches.view(3, self.patch_size, self.patch_size, num_patches)
        sharp_patches = sharp_patches.view(3, self.patch_size, self.patch_size, num_patches)

        blurry_patches = blurry_patches.permute(3, 0, 1, 2)
        sharp_patches = sharp_patches.permute(3, 0, 1, 2)

        return blurry_patches, sharp_patches

    def original_shape(self, idx):
        blurry_image_path = self.blurry_image_paths[idx]
        sharp_image_path = self.sharp_image_paths[idx]

        blurry_image = Image.open(blurry_image_path)
        sharp_image = Image.open(sharp_image_path)

        blurry_image = transforms.ToTensor()(blurry_image)
        sharp_image = transforms.ToTensor()(sharp_image)

        return blurry_image.shape, sharp_image.shape
