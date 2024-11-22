import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Image parameters (same as in training file)
img_width, img_height = 224, 224

# Prediction function
def predict_image_class(image_path, model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = "Fake" if prediction[0] > 0.5 else "Real"
    print(f"Predicted class: {predicted_class}")
    print(f"Prediction probability: {prediction[0][0]:.4f}")

# Example usage: Predict a new image
image_file = '/path/to/image.jpg'  # Replace with the image path you want to classify
model_path = './model_checkpoints/vgg16_final_model.keras'
predict_image_class(image_file, model_path)
