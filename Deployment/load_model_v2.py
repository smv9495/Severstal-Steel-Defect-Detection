import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from unet_from_segmentation_model_v3 import Unet, Unet_plus_plus
import os

def get_classification_model():
    model = ResNet50(include_top=False, input_shape=(256, 400, 3))
    flat1 = GlobalAveragePooling2D()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu', kernel_initializer='he_normal')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.load_weights(os.path.join('weights', 'ResNet50_final_weights.h5'))
    return model

def get_segmentation_model():
    unet_model = Unet(
                        backbone_name='efficientnetb4',
                        input_shape=(256, 400, 3),
                        classes=5,
                        activation='softmax',
                        weights=None,
                        encoder_weights='imagenet',
                        encoder_freeze=False,
                        encoder_features='default',
                        decoder_block_type='upsampling',
                        decoder_filters=(256, 128, 64, 32, 16),
                        decoder_use_batchnorm=True,
                )

    unet_model.load_weights(os.path.join('weights', 'final-segmentation-weights.h5'))
    return unet_model