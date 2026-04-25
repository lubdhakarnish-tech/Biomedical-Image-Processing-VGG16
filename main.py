# =====================================
# VGG16 Image Classification Project
# =====================================

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# =========================
# DATA LOADING
# =========================
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# =========================
# MODEL 1: FROM SCRATCH
# =========================
def build_scratch_model():
    model = Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# =========================
# MODEL 2: PRETRAINED VGG16
# =========================
def build_pretrained_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

    # Freeze layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# =========================
# MODEL 3: FINE-TUNED VGG16
# =========================
def build_finetune_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

    # Freeze most layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# =========================
# TRAINING FUNCTION
# =========================
def train_model(model, name):
    print(f"\nTraining {name}...\n")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    return history

# =========================
# PLOT FUNCTION
# =========================
def plot_history(history, title):
    plt.figure()

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # Model 1
    scratch_model = build_scratch_model()
    history1 = train_model(scratch_model, "VGG16 Scratch")
    plot_history(history1, "Scratch Model")

    # Model 2
    pretrained_model = build_pretrained_model()
    history2 = train_model(pretrained_model, "Pretrained VGG16")
    plot_history(history2, "Pretrained Model")

    # Model 3
    finetune_model = build_finetune_model()
    history3 = train_model(finetune_model, "Fine-tuned VGG16")
    plot_history(history3, "Fine-tuned Model")
