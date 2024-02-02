import torch

folder = "/content/drive/MyDrive/Training_model"
checkpoint_path = f"{folder}/Denoise_epoch_47.pth"
device = torch.device("cpu")
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

# Identify and remove keys containing "sca"
keys_to_remove = [key for key in checkpoint["model_state_dict"] if "sca" in key]
for key in keys_to_remove:
    del checkpoint["model_state_dict"][key]

# Save the modified checkpoint
torch.save({
    'model_state_dict': checkpoint["model_state_dict"],
    'optimizer_state_dict': checkpoint['optimizer_state_dict'],
}, f"{folder}/Denoise_epoch_47_no_sca.pth")