import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set dataset path
data_dir = r"C:\Users\deLL\Downloads\cat-breeds"
image_size = (224, 224)
batch_size = 32
seed = 123

# Check and print class distribution
root = Path(data_dir)
class_counts = {folder.name: len(list(folder.glob('*'))) for folder in root.iterdir() if folder.is_dir()}
print("📊 Class Distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")

if len(class_counts) < 2:
    raise ValueError("⚠️ ERROR: Dataset must contain at least 2 classes in separate folders!")

# Load datasets with validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Save class names
class_names = train_ds.class_names
print(f"✅ Found {len(class_names)} classes: {class_names}")
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

# Normalize and prepare datasets
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# Define transfer learning model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze base model

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save model
model.save('cat_breed_classifier.h5')
print("✅ Model saved as 'cat_breed_classifier.h5'")

# Plot training curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.show()
