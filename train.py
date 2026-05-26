import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import os

# --- 1. CONFIGURATION ---
DATASET_DIR = r"D:\FogMap\SMOG4000" # Main folder containing train, valid, and test
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("🚀 Initializing Advanced FOGMAP Pipeline...")

# --- 2. LOAD DATASET ---
# Pointing directly to the pre-split subfolders to fix the ValueError
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")

print("📂 Loading datasets from split directories...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Optimize datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. THE SECRET WEAPON: AGGRESSIVE DATA AUGMENTATION ---
# This stops the model from confusing windshield glare with fog
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(factor=0.3), # Teaches the model to ignore sun glare
    layers.RandomContrast(factor=0.3)    # Teaches the model to ignore washed-out windshields
], name="aggressive_augmentation")

# --- 4. UPGRADED ARCHITECTURE: EfficientNetB0 ---
# EfficientNet is much better at separating background haze from foreground objects
base_model = applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model for initial training
base_model.trainable = False

# Build the custom FOGMAP head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x) # 40% Dropout prevents the model from memorizing specific pixels
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

# --- 5. PHASE 1: WARM-UP TRAINING ---
print("\n🔥 PHASE 1: Training Custom Head...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks to save the best version and stop if it stops learning
early_stop = callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stop]
)

# --- 6. PHASE 2: DEEP FINE-TUNING ---
print("\n🔬 PHASE 2: Deep Fine-Tuning (Teaching it what 'Haze' really is)...")
base_model.trainable = True

# We only unfreeze the top 30 layers of EfficientNet to prevent destroying pre-trained edge detection
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a MICRO learning rate so we don't shock the weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Save the absolute best model
checkpoint = callbacks.ModelCheckpoint(
    "fogmap_efficientnet_v2.keras", 
    save_best_only=True, 
    monitor='val_accuracy'
)

history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stop, checkpoint]
)

print("\n✅ Training Complete! The robust model is saved as 'fogmap_efficientnet_v2.keras'")
