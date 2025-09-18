from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load model
model = load_model("cnn_cats_dogs.h5")

# Test generator
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_directory("data/test", target_size=(150,150), batch_size=32, class_mode='binary')

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
