from patchify import patchify, unpatchify
import numpy as np
import matplotlib.pyplot as plt
from load_model_v2 import get_classification_model, get_segmentation_model
import cv2
import os

def read_image(image, flag='IMREAD_UNCHANGED'):
    #return numpy array
    if flag == 'IMREAD_UNCHANGED':
        return cv2.imread(image, cv2.IMREAD_UNCHANGED)
    elif flag == 'IMREAD_GRAYSCALE':
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        raise Exception("Please provide valid flag")

def image_to_patch(sample_image, patch_size=(256,400,3), step=400, normalize=False):
    #divide input image into patches of equal size
    img_patches = patchify(sample_image, patch_size, step)
    img_patches = np.squeeze(img_patches, (0,2))
    if normalize:
        img_patches = img_patches/255.
    return img_patches

def preprocessing_classification(img_patches, classifier):
    #predict if defect is present or not
    y_hat = classifier.predict(img_patches)
    y_label = (y_hat>0.3).astype(int)
    return y_label

def preprocessing_segmentation(img_patches, model, label):
    #perform segmentation and return segmented mask
    n = label.flatten().shape[0]
    predicted_patches = model.predict(img_patches) * label.reshape((n,1,1,1))
    predicted_patches = np.expand_dims(predicted_patches, (0,2))
    reconstructed_patch = unpatchify(predicted_patches, (256,1600,5))
    reconstructed_patch = np.argmax(reconstructed_patch, axis=-1)
    return reconstructed_patch

def get_encoded_pixels(recovered_mask):
    #return encoded pixel values for kaggle submission
    encoded_pixel_recovered = []
    unique_pixels = np.unique(recovered_mask)
    for defect in range(1,5):
        if defect in unique_pixels:
            recovered_pixels = np.where(recovered_mask.T.flatten()==defect)[0]+1
            array = np.where((recovered_pixels[1:] - recovered_pixels[:-1])!=1)[0]
            start_array = np.concatenate([[recovered_pixels[0]], recovered_pixels[array+1]])
            end_array = np.concatenate([recovered_pixels[array], [recovered_pixels[-1]]])
            offset = end_array - start_array + 1
            encoded_pixel_recovered_per_class = ' '.join([f'{start_array[i]} {offset[i]}' for i in range(offset.shape[0])])
            encoded_pixel_recovered.append(encoded_pixel_recovered_per_class)
        else:
            encoded_pixel_recovered.append('')
    return encoded_pixel_recovered

def combined_inference(sample_image, classifier, segmentation_model):
    #combine all the above process into single inference function
    img_patches = image_to_patch(sample_image)
    label = preprocessing_classification(img_patches, classifier)
    if label.sum() == 0:
        return None
    else:
        recovered_mask = preprocessing_segmentation(img_patches, segmentation_model, label)
    return recovered_mask

if __name__ == '__main__':
    classification_model = get_classification_model()
    segmentation_model = get_segmentation_model()
    image_path = 'D:\\AppliedAi\\self_case_study_2\\Deployment\\InputData\\train_images\\0a4ad45a5.jpg'
    sample_image = read_image(image_path)
