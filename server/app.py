from flask import Flask, request, send_file, jsonify
import os
from Utils import inpaint_inference, deblur_inference, denoise_inference, handle_base64
import matplotlib.pyplot as plt
from flask_cors import CORS
import base64
import numpy as np
from datetime import datetime


app = Flask(__name__)
CORS(app)


@app.route("/")
def welcome():
    return "Welcome to image restoration app"


@app.route("/denoise", methods=["POST", "GET"])
def denoise():
    base64_string = request.form["base64_string"]
    print(base64_string)
    image_path = handle_base64.decode(base64_string)
    print("denoising... Start")
    output_image_noisy, output_image_clear = denoise_inference.predict(image_path)
    print("denoising... Done")
    print(output_image_clear.shape)
    # # response.headers.add('Access-Control-Allow-Origin', '*')
    denoised_image_path = "F:/machine-learning/Image_restoration_UI/database/denoised.jpg"
    plt.imsave(denoised_image_path, output_image_clear)
    print(jsonify({'path': "database/denoised.jpg"}))
    return jsonify({'path': "database/denoised.jpg"})


@app.route("/deblur", methods=["POST", "GET"])
def deblur():
    base64_string = request.form["base64_string"]
    print(base64_string)
    image_path = handle_base64.decode(base64_string)
    print("deblur... Start")
    output_image_blurry, output_image_clear = deblur_inference.predict(image_path)
    print("deblur... Done")
    deblurred_image_path = "F:/machine-learning/Image_restoration_UI/database/deblurred.jpg"
    plt.imsave(deblurred_image_path, output_image_clear)
    print(jsonify({'path': "database/deblurred.jpg"}))
    return jsonify({'path': "database/deblurred.jpg"})


@app.route("/inpaint", methods=["POST", "GET"])
def inpaint():
    base64_string = request.form["base64_string"]
    print(base64_string)
    image_path = handle_base64.decode(base64_string)
    print("Inpainting... Start")
    print(image_path[:-8])
    inpaint_inference.run_inference(image_path[:-8], )
    print("Inpainting... Done")
    print(jsonify({'path': "database/output/final_output/old.png"}))
    return jsonify({'path': "database/output/final_output/old.png"})


if __name__ == "__main__":
    app.run(port=5000)


