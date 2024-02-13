from image_restoration_web_app.Utils.addToExcelsheet import add_to_excelsheet
import torch
import torch.nn as nn
from image_restoration_web_app.Archs.NafnetArch import NAFNet

def train(gopro_dataloader_train, folder, checkpoint_path, excel_file, progress_file, device="cuda", lr=0.0001):
    device = torch.device(device)

    img_channel = 3
    width = 32

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)

    criterion = nn.MSELoss()
    lr = lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device(device)
    epoch = int(checkpoint_path.split("/")[-1].split("_")[-1][:-4])
    print(epoch)
    i = 0
    num_epochs = epoch + 2
    with open(progress_file, "w") as file:
        for epoch in range(epoch + 1, num_epochs):
            i = 0
            batch_loss = []
            for batch_no, (noisy_images, gt_images) in enumerate(gopro_dataloader_train):
                patches_loss = []
                for patch_index in range(noisy_images[0].size(0)):
                    noisy_image = noisy_images[:, patch_index].to(device)
                    gt_image = gt_images[:, patch_index].to(device)
                    optimizer.zero_grad()
                    outputs = model(noisy_image)
                    loss = criterion(outputs, gt_image)
                    loss.backward()
                    optimizer.step()
                    patches_loss.append(loss.item())
                    progress_str = f"Epoch [{epoch}], Step [{batch_no+1}/{len(gopro_dataloader_train)}], Patch [{patch_index+1}/{noisy_images[0].size(0)}], Loss: {loss.item()}"
                    print(progress_str)
                    file.write(progress_str + "\n")
                batch_loss.append(sum(patches_loss) / len(patches_loss))
            i += 1
            if i % 200 == 0 and i != 0:
                    torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, f"{folder}/GoPro_epoch_{epoch}.pth")
                    progress_str = f"Epoch [{epoch}], Epoch Loss: {sum(batch_loss) / len(batch_loss)}"
                    print(progress_str)
                    file.write(progress_str + "\n")
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{folder}/GoPro_epoch_{epoch}.pth")
            Epoch_loss = sum(batch_loss) / len(batch_loss)
            progress_str = f"Epoch [{epoch}], Epoch Loss: {Epoch_loss}"
            print(progress_str)
            add_to_excelsheet(excel_file,epoch,0, Epoch_loss,lr)
            file.write(progress_str + "\n")
    file.close()