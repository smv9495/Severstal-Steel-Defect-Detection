import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

palette = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]

#code is adopted from below link and modified as per need
#https://www.kaggle.com/code/go1dfish/clear-mask-visualization-and-simple-eda
def show_mask_image(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    name = os.path.basename(image_path)
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours((mask==ch+1).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palette[ch], 2)
    ax.set_title(name)
    ax.imshow(img)
    plt.show()

def save_mask_image(img, mask, name):
    for ch in range(4):
        contours, _ = cv2.findContours((mask==ch+1).astype(np.uint8),
                                        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palette[ch], 2)
    output_img_path = os.path.join('static', name)
    plt.imsave(output_img_path, img)

if __name__ == '__main__':
    image_path = os.path.join('ProcessedData','Steel_Defect_Detection',
                              'input_data','train_images',
                              'defective','0a4ad45a5.jpg')
    mask_path = os.path.join('ProcessedData','Steel_Defect_Detection',
                              'input_data','train_mask',
                              '0a4ad45a5.png')
    show_mask_image(image_path, mask_path)