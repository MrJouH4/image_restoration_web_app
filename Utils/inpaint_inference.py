import os.path
from subprocess import Popen, PIPE
from PIL import Image

def run_inference(image_path, output_folder, gpu=-1, with_scratch=True):
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
        final_output = os.path.join(output_folder, "final_output")
        files = os.listdir(final_output)
        image_files = [f for f in files if os.path.isfile(os.path.join(final_output, f)) and f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        image_path = os.path.join(final_output, image_files[0])
        inpainted_image = Image.open(image_path)
        print("Inference completed successfully.")
        return inpainted_image



# run_inference("C:/Users/Jou/Desktop/test_image", "C:/Users/Jou/Desktop/output")
