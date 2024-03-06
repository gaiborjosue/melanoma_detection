from tensorflow.keras.preprocessing.image import load_img, img_to_array
import imageio
import numpy as np

def load_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Load the image and resize it to 224x224 pixels
    img = img_to_array(img)  # Convert the image to a numpy array
    return img

from tensorflow.keras.applications.resnet import preprocess_input

# Load the image
image = load_image("Meshtest/melanoma_detection/ISIC_0000036.jpg")

# Preprocess the image
image = preprocess_input(image)

image_to_save = image - np.min(image)  # Shift values to range 0-...
image_to_save /= np.max(image_to_save)  # ...-1
image_to_save *= 255  # Scale values to range 0-255

# Convert the image back to RGB
image_to_save = image_to_save[..., ::-1]

# Save the image
imageio.imsave('preprocessed_image.jpg', image_to_save.astype(np.uint8))