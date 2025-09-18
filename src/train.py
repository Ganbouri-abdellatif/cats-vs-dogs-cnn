import os
import data_utils  # import the module itself
import utils       # import utils for plotting
from model import build_cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
RAW_DIR = "/home/cipher/Documents/programming/cats-vs-dogs/cats-vs-dogs-dataset"
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Create train/valid/test folders for each class
data_utils.create_dirs([os.path.join(DATA_DIR, split, cls)
                        for split in ["train", "valid", "test"]
                        for cls in ["Cat", "Dog"]])

# Split data automatically
data_utils.split_data(os.path.join(RAW_DIR, "Cat"),
                      os.path.join(TRAIN_DIR, "Cat"),
                      os.path.join(VALID_DIR, "Cat"),
                      os.path.join(TEST_DIR, "Cat"))

data_utils.split_data(os.path.join(RAW_DIR, "Dog"),
                      os.path.join(TRAIN_DIR, "Dog"),
                      os.path.join(VALID_DIR, "Dog"),
                      os.path.join(TEST_DIR, "Dog"))

# Image generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen   = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_gen.flow_from_directory(
    VALID_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build and train model
model = build_cnn_model()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Plot training curves
utils.plot_history(history)

# Save model
model.save("cnn_cats_dogs.h5")
print("Model saved as cnn_cats_dogs.h5")
