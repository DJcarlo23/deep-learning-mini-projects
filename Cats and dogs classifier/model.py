from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img
from pathlib import Path

import tensorflow.keras as keras
import numpy as np

class Model():
    def __init__(self):
        self.base_model = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))
        
        self.main_model = keras.models.load_model(Path('cats_and_dogs_model.h5'))

    def load_data(self, image_path):
        image = load_img(image_path, target_size=(150, 150))
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1, 150, 150, 3)

        return img

    def classify_image(self, image_path):

        img_array = self.load_data(image_path)

        results = self.base_model.predict(img_array)
        results = np.reshape(results, (1, 4*4*512))

        proba = self.main_model.predict(results)

        return proba
