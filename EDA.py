import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# 1. Base Directory
# Since your terminal showed you are running from D:\FogMap, we can use the absolute path
base_dir = r"D:\FogMap\SMOG4000"

# 2. Match exact lowercase folder names
train_dir = os.path.join(base_dir, 'train')

# 3. Match exact class folder names ('clear' and 'smog')
clear_dir = os.path.join(train_dir, 'clear')
smog_dir = os.path.join(train_dir, 'smog')

# Print statements to help you debug and verify paths
print(f"Looking for clear images in: {clear_dir}")
print(f"Looking for smog images in: {smog_dir}")

# Grab a random sample of image filenames
clear_images = [os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if f.endswith(('jpg', 'png', 'jpeg', 'JPG'))]
smog_images = [os.path.join(smog_dir, f) for f in os.listdir(smog_dir) if f.endswith(('jpg', 'png', 'jpeg', 'JPG'))]

print(f"Found {len(clear_images)} clear images and {len(smog_images)} smog images.")

# Let's plot 3 random Clear images and 3 random Smog images
plt.figure(figsize=(15, 6))
for i in range(3):
    # Plot Clear Images
    plt.subplot(2, 3, i + 1)
    img = mpimg.imread(random.choice(clear_images))
    plt.imshow(img)
    plt.title(f"Clear Image {img.shape}")
    plt.axis('off')

    # Plot Smog Images
    plt.subplot(2, 3, i + 4)
    img = mpimg.imread(random.choice(smog_images))
    plt.imshow(img)
    plt.title(f"Smog Image {img.shape}")
    plt.axis('off')

plt.tight_layout()
plt.show()