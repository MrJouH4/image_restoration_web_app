import random
import matplotlib.pyplot as plt


def display_image(dataset):
    random_img = random.randint(0, len(dataset) - 1)
    random_patch = random.randint(0, len(dataset[0][0]) - 1)
    sample = dataset[random_img]
    noise_image = sample[0][random_patch]
    hr_image = sample[1][random_patch]

    noisy_image = noise_image.permute(1, 2, 0).cpu().numpy()
    ground_truth_image = hr_image.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(noisy_image)
    axes[0].set_title("Noisy Image")
    axes[0].axis('off')

    axes[1].imshow(ground_truth_image)
    axes[1].set_title("Ground Truth Image")
    axes[1].axis('off')

    plt.show()
