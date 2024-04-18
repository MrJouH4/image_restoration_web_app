from flask import Flask, request
import os
from Utils import inpaint_inference, deblur_inference

app = Flask(__name__)


@app.route("/")
def welcome():
    return "Welcome to image restoration app"

@app.route("/inpaint", methods=["POST", "GET"])
def inpaint():
    input_path = request.form["input_path"]
    output_path = os.path.join(input_path, "output")
    os.mkdir(output_path)
    print("Inpainting... Start")
    inpaint_inference.run_inference(input_path, output_path)
    print("Inpainting... Done")
    return '', 200


if __name__ == "__main__":
    app.run(port=5000)