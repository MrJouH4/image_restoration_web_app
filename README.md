
# Image Restoration Web App

an Image Restoration Web App capable of denoising, deblurring Using NAFNet Model but we edited Simplified Channel Attention Module to be Multi-head Transposed Attention, and inpainting old images Using Microsoft's deep latent space translation pretrained model. 

## Features

-   **Denoising:** Removes noise from images.
-   **Deblurring:** Enhances blurred images, restoring sharpness and clarity.   
- **Inpainting:** Fills in missing or damaged areas of images seamlessly.

## Usage

To run the web app locally, follow these steps:

1.  Clone this repository to your local machine.
2.  Navigate to the project directory.
3.  Install the necessary dependencies by running.
    
    ```
    pip install -r requirements.txt
    mkdir database
    ``` 
    Download this repo and follow the instructions:
    [microsoft/Bringing-Old-Photos-Back-to-Life: Bringing Old Photo Back to Life (CVPR 2020 oral) (github.com)](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)
    in this cloned repo folder run this commands
    ```
    mkdir checkpoints
    cd checkpoints
    gdown 1vZ5w3asvVElzPn4UrHfmFkPRkACSTAlb
    gdown 12wWjV9tkmlLL7m12qDiR9_BOkX2pBF52
    ```
    
4.  Run the web app:

    `python server/app.py` 
    
5. Clone this repo in another directory
```
git clone https://github.com/MrJouH4/image-restoration-UI
```
6. open terminal in this directory and run these commands
```
npm install
npm run dev
```
7. Open this on your browser => http://localhost:5173/

8. Enjoy Restoring your Old Images 🥰



## Download

If you can run gdown commands for checkpoint downloading. You can download the pre-trained models and checkpoints from the following links:

-   [Denoising Model Checkpoint](https://chat.openai.com/c/8791aa7c-19d6-4568-a283-c66df54f7f9b#)
-   [Deblurring Model Checkpoint](https://chat.openai.com/c/8791aa7c-19d6-4568-a283-c66df54f7f9b#)
-   [Inpainting Model Checkpoint](https://chat.openai.com/c/8791aa7c-19d6-4568-a283-c66df54f7f9b#)

## Credits

- [megvii-research/NAFNet: The state-of-the-art image restoration model without nonlinear activation functions. (github.com)](https://github.com/megvii-research/NAFNet)
- [microsoft/Bringing-Old-Photos-Back-to-Life: Bringing Old Photo Back to Life (CVPR 2020 oral) (github.com)](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)
