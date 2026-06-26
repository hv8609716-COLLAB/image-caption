import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import requests
from io import BytesIO

# Load pretrained model
model = InceptionV3(weights='imagenet')

# Load image from URL
# The previous URLs resulted in 403 Forbidden errors (access denied).
# Changing to a new, publicly accessible image URL from picsum.photos.
img_path = "" # Using a reliable public image source
response = requests.get(img_path)

# Check if the request was successful
if response.status_code != 200:
    raise requests.exceptions.RequestException(f"Failed to retrieve image from {img_path}. Status code: {response.status_code}")

try:
    img_bytes = BytesIO(response.content)
    img = Image.open(img_bytes).convert("RGB").resize((299, 299))
    # Create a separate image object for display to avoid resizing the original for display
    img_display = Image.open(BytesIO(response.content)) # Re-read for display to avoid PIL's internal state issues
except UnidentifiedImageError:
    raise UnidentifiedImageError(f"Cannot identify image file from {img_path}. The downloaded content might not be a valid image.")

img = np.array(img)

# Fix RGBA issue (if any)
if img.shape[-1] == 4:
    img = img[:, :, :3]

img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Predict
pred = model.predict(img)
labels = decode_predictions(pred)[0]

# Show result
caption = labels[0][1]

print("Caption: ", caption)

# Display image
plt.imshow(img_display)
plt.title(caption)
plt.axis('off')
plt.show()
