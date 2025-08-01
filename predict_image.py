import tensorflow as tf 
import numpy as np
from PIL import Image
import os

model_path = 'copra_classifier/models/copra_model.keras'
model = tf.keras.models.load_model(model_path)
print(f'Model loaded from: {model_path}')

class_names = ['overcooked', 'raw', 'standard']

image_path = input('enter image file path to predict: ').strip()

if not os.path.exists(image_path):
    print('image not found, input correct path')
    exit()
    
img = Image.open(image_path).resize((180, 180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = 100 * np.max(predictions)

print(f"""
    \n Prediction Result:
    \n Class: {predicted_class}
    \m Confidence: {confidence:.2f}%
    """)