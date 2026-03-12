import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
# Point this to the tricky nature fog image (or any image you want to test)
IMAGE_PATH = r"D:\FogMap\test_clear.jpg" 

# UPGRADED: Now pointing to our fine-tuned model!
MODEL_PATH = r"D:\FogMap\smog_classifier_finetuned.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Clear', 'Smog'] # 0 is Clear, 1 is Smog

# ==========================================
# 2. Load the Model
# ==========================================
print("Loading fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# ==========================================
# 3. Load and Preprocess the Image
# ==========================================
# Load the image and resize it to match the training data format
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)

# Convert the image to a numpy array (pixel values 0-255)
img_array = image.img_to_array(img)

# The model expects a "batch" of images, not just one. 
# We use expand_dims to make it look like a batch of 1 image: shape (1, 224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# 4. Make the Prediction
# ==========================================
# The model outputs a probability between 0 and 1
prediction = model.predict(img_array, verbose=0)[0][0]

# Determine the class based on a 0.5 threshold
if prediction > 0.5:
    predicted_class = "Smog"
    confidence = prediction * 100
else:
    predicted_class = "Clear"
    confidence = (1 - prediction) * 100

print(f"Result: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

# ==========================================
# 5. Display the Result Visually
# ==========================================
plt.figure(figsize=(6, 6))
# Load original image for displaying (without the 224x224 squeeze)
display_img = image.load_img(IMAGE_PATH)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class} ({confidence:.1f}%)", fontsize=16, fontweight='bold')
plt.axis('off')
plt.show()