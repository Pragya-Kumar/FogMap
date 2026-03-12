import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 1. Define Paths 
# ==========================================
base_dir = r"D:\FogMap\SMOG4000"

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# ==========================================
# 2. Hyperparameters & Dataset Loading
# ==========================================
BATCH_SIZE = 32
IMG_SIZE = (224, 224) # Resizing all those different sized images here!
EPOCHS = 15

print("Loading datasets...")

train_dataset = image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary' 
)

val_dataset = image_dataset_from_directory(
    val_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

test_dataset = image_dataset_from_directory(
    test_dir,
    shuffle=False, # Keep false so labels match our predictions later
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

# ==========================================
# 3. Build the Model (Transfer Learning)
# ==========================================
# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

# Preprocessing for MobileNetV2 (scales pixels to [-1, 1])
preprocess_input = applications.mobilenet_v2.preprocess_input

# Load pre-trained MobileNetV2 base
base_model = applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                      include_top=False,
                                      weights='imagenet')
base_model.trainable = False # Freeze base layers

# Construct final model
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x) 
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# ==========================================
# 4. Compile and Train
# ==========================================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

print("\nStarting Training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

# ==========================================
# 5. Evaluate on Test Data
# ==========================================
print("\nEvaluating on Test Dataset...")
loss, accuracy, recall = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Recall: {recall*100:.2f}%")

# ==========================================
# 6. Plotting Results (Graphs & Confusion Matrix)
# ==========================================
# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# Generate Confusion Matrix
print("\nGenerating Confusion Matrix...")
y_true = []
y_pred_probs = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy().flatten())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds.flatten())

y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save('smog_classifier_model.keras')
print("\nModel saved as 'smog_classifier_model.keras'")