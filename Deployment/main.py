from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import numpy as np
from PIL import Image, ImageShow
import os
import cv2
from processing import combined_inference
from load_model_v2 import get_classification_model, get_segmentation_model
import matplotlib.pyplot as plt
from mask_visualization import save_mask_image

app = Flask(__name__)
classification_model = get_classification_model()
segmentation_model = get_segmentation_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display_result')
def display():
    return render_template('display.html')

@app.route('/display_no_defect_found')
def display_no_defect_found():
    return render_template('display_no_defect_found.html')

@app.route('/processor', methods=['POST'])
def preprocessing():
    global classification_model, segmentation_model
    f = request.files['image_file']
    img = Image.open(f)
    img = np.asarray(img)

    # In case of grayScale images the len(img.shape) == 2
    if len(img.shape) > 2 and img.shape[2] == 4:
        #convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    in_img = Image.fromarray(img)
    in_img.save('static/input_img.png')

    output = combined_inference(img, classification_model, segmentation_model)

    if output is not None:
        save_mask_image(img.copy(), output, 'output_img.png')
        return redirect(url_for('display'))

    return redirect(url_for('display_no_defect_found'))
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080, debug=True)