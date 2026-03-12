import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# ==========================================
# 1. Load the Datasets
# ==========================================
base_dir = r"D:\FogMap\SMOG4000"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("Loading data for fine-tuning...")
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='binary')
val_dataset = image_dataset_from_directory(val_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='binary')

# ==========================================
# 2. Load the Saved Model
# ==========================================
print("Loading the existing 97% accuracy model...")
model = tf.keras.models.load_model('smog_classifier_model.keras')

# ==========================================
# 3. Unfreeze the Base Model Layers
# ==========================================
# Look through the layers to find the MobileNetV2 feature extractor
base_model = None
for layer in model.layers:
    if layer.name.startswith('mobilenetv2'):
        base_model = layer
        break

if base_model is None:
    raise ValueError("Could not find the MobileNetV2 base model. Check layer names.")

# Unfreeze the base model
base_model.trainable = True

# MobileNetV2 has 154 layers. We will freeze the first 100 (basic edges/shapes)
# and ONLY fine-tune the last 54 layers (complex patterns like haze).
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Total layers in base model: {len(base_model.layers)}")
print(f"Fine-tuning from layer {fine_tune_at} onwards...")

# ==========================================
# 4. Re-Compile with a MICRO Learning Rate
# ==========================================
# We divide our original learning rate by 10 or 100 so we don't wreck the pre-trained weights
base_learning_rate = 0.0001
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
              metrics=['accuracy'])

model.summary()

# ==========================================
# 5. Start Fine-Tuning
# ==========================================
FINE_TUNE_EPOCHS = 10 # We don't need many epochs for fine-tuning

print("\nStarting Fine-Tuning Phase...")
history_fine = model.fit(
    train_dataset,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_dataset
)

# Save the newly upgraded model!
model.save('smog_classifier_finetuned.keras')
print("\nFine-tuned model saved as 'smog_classifier_finetuned.keras'")

# Plot the fine-tuning progress
plt.figure(figsize=(8, 6))
plt.plot(history_fine.history['accuracy'], label='Fine-Tune Train Acc')
plt.plot(history_fine.history['val_accuracy'], label='Fine-Tune Val Acc')
plt.title('Fine-Tuning Accuracy')
plt.legend()
plt.show()