import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

class SIDDDataset(Dataset):
    def __init__(self, root_dir, patch_size):
        super().__init__()

        self.root_dir = root_dir
        self.patch_size = patch_size

        self.noisy_image_paths = []
        self.ground_truth_image_paths = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            for img in os.listdir(folder_path):
                img_parts = img.split("_")
                if img_parts[1].startswith("G"):
                    self.ground_truth_image_paths.append(os.path.join(root_dir, folder, img))
                if img_parts[1].startswith("N"):
                    self.noisy_image_paths.append(os.path.join(root_dir, folder, img))

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, idx):
        noisy_image_path = self.noisy_image_paths[idx]
        ground_truth_image_path = self.ground_truth_image_paths[idx]

        noisy_image = Image.open(noisy_image_path)
        ground_truth_image = Image.open(ground_truth_image_path)

        noisy_image = transforms.ToTensor()(noisy_image)
        ground_truth_image = transforms.ToTensor()(ground_truth_image)

        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        noisy_patches = unfold(noisy_image)
        ground_truth_patches = unfold(ground_truth_image)

        num_patches = noisy_patches.shape[-1]
        noisy_patches = noisy_patches.view(3, self.patch_size, self.patch_size, num_patches)
        ground_truth_patches = ground_truth_patches.view(3, self.patch_size, self.patch_size, num_patches)

        noisy_patches = noisy_patches.permute(3, 0, 1, 2)
        ground_truth_patches = ground_truth_patches.permute(3, 0, 1, 2)

        return noisy_patches, ground_truth_patches

    def original_shape(self, idx):
        noisy_image_path = self.noisy_image_paths[idx]
        ground_truth_image_path = self.ground_truth_image_paths[idx]

        noisy_image = Image.open(noisy_image_path)
        ground_truth_image = Image.open(ground_truth_image_path)

        noisy_image = transforms.ToTensor()(noisy_image)
        ground_truth_image = transforms.ToTensor()(ground_truth_image)

        return noisy_image.shape, ground_truth_image.shape