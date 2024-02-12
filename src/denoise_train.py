from image_restoration_web_app.Utils.addToExcelsheet import add_to_excelsheet
import random
import torch
import torch.nn as nn
from Archs.NafnetArch import NAFNet


def train(sidd_dataloader_train, div2k_dataloader_train, folder, checkpoint_path,
          excel_file, progress_file, device="cuda", lr=0.0001):

    device = torch.device(device)

    img_channel = 3
    width = 32

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device(device)
    epoch = int(checkpoint_path.split("/")[-1].split("_")[-1][:-4])
    print(epoch)
    num_epochs = epoch + 2
    with open(progress_file, "w") as file:
        for epoch in range(epoch + 1, num_epochs):
            i = 0
            batch_loss = []
            for batch_no, ((sidd_noisy_images, sidd_gt_images), (div2k_noisy_images, div2k_gt_images)) in enumerate(
                    zip(sidd_dataloader_train, div2k_dataloader_train)):
                sidd_patches_loss = []
                div2k_patches_loss = []
                for patch_index in range(div2k_noisy_images[0].size(0)):
                    # SIDD
                    random_index = random.randint(0, (sidd_noisy_images[0].size(0)) - 1)
                    sidd_noisy_image = sidd_noisy_images[:, random_index].to(device)
                    sidd_gt_image = sidd_gt_images[:, random_index].to(device)
                    optimizer.zero_grad()
                    sidd_outputs = model(sidd_noisy_image)
                    sidd_loss = criterion(sidd_outputs, sidd_gt_image)
                    sidd_loss.backward()
                    optimizer.step()
                    sidd_patches_loss.append(sidd_loss.item())
                    # DIV2K
                    div2k_noisy_image = div2k_noisy_images[:, patch_index].to(device)
                    div2k_gt_image = div2k_gt_images[:, patch_index].to(device)
                    optimizer.zero_grad()
                    div2k_outputs = model(div2k_noisy_image)
                    div2k_loss = criterion(div2k_outputs, div2k_gt_image)
                    div2k_loss.backward()
                    optimizer.step()
                    div2k_patches_loss.append(div2k_loss.item())

                    progress_str = f"Epoch [{epoch}], Step [{batch_no + 1}/{len(sidd_dataloader_train)}], Patch [{patch_index + 1}/{min(sidd_noisy_images[0].size(0), div2k_noisy_images[0].size(0))}], SIDD Loss: {sidd_loss.item()}, DIV2K Loss: {div2k_loss.item()}"
                    print(progress_str)
                    file.write(progress_str + "\n")

                sidd_batch_loss = sum(sidd_patches_loss) / len(sidd_patches_loss)
                div2k_batch_loss = sum(div2k_patches_loss) / len(div2k_patches_loss)
                batch_loss.append((sidd_batch_loss, div2k_batch_loss))
                i += 1
                if i % 20 == 0 and i != 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{folder}/Denoise_epoch_{epoch}.pth")
                    sidd_epoch_loss = sum([s[0] for s in batch_loss]) / len(batch_loss)
                    div2k_epoch_loss = sum([s[1] for s in batch_loss]) / len(batch_loss)
                    progress_str = f"Epoch [{epoch}], SIDD Epoch Loss: {sidd_epoch_loss}, DIV2K Epoch Loss: {div2k_epoch_loss}"
                    print(progress_str)
                    file.write(progress_str + "\n")
            add_to_excelsheet(excel_file, epoch, sidd_epoch_loss, div2k_epoch_loss, lr)
    file.close()
