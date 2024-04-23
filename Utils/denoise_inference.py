import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Archs.NafnetArch import NAFNet


def predict(input_image_path):
    checkpoint_path = "../checkpoints/Denoise_epoch_68.pth"

    img_channel = 3
    width = 32
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    device = torch.device("cpu")

    predict_model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)
    optimizer = torch.optim.Adam(params=predict_model.parameters(), lr=0.0001)


    checkpoint = torch.load(checkpoint_path, map_location=device)
    predict_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    predict_model.eval()

    noisy_image = Image.open(input_image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    noisy_image = transform(noisy_image).to(device)


    with torch.no_grad():
        output_chunk = predict_model(torch.unsqueeze(noisy_image, 0))

    image_np1 = noisy_image.permute(1, 2, 0).cpu().numpy()
    image_np2 = output_chunk[0].permute(1, 2, 0).cpu().numpy()
    image_np2 = (image_np2 - image_np2.min()) / (image_np2.max() - image_np2.min())
    return image_np1, image_np2


if __name__ == "__main__":
    noisy, tested = predict("C:/Users/Jou/Desktop/fcai.png")
    # noisy, tested = predict("../database/old.png")

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(noisy)
    axes[0].set_title("Noisy Image")
    axes[0].axis('off')

    axes[1].imshow(tested)
    axes[1].set_title("Tested Image")
    axes[1].axis('off')
    plt.show()