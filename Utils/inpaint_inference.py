import os.path
from subprocess import Popen, PIPE
from PIL import Image
import io
import base64
import os


def run_inference(image_path, output_folder="F:/machine-learning/Image_restoration_UI/database/output", gpu=-1, with_scratch=True):
    command = [
        "python", "../Bringing-Old-Photos-Back-to-Life/run.py",
        "--input_folder", str(image_path),
        "--output_folder", str(output_folder),
        "--GPU", str(gpu),
    ]
    if with_scratch:
        command.append("--with_scratch")

    process = Popen(command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("Error running inference:")
        print(stderr.decode())
    else:
        print("Inference completed successfully.")


if __name__ == "__main__":
    run_inference("C:/Users/Jou/Desktop/test_image", "C:/Users/Jou/Desktop/denoise_output")