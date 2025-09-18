import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("cnn_cats_dogs.h5", compile=False)
class_names = ["Cat", "Dog"]

def predict_image(img_path):
    """Predict the class of a single image."""
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)
    label = class_names[np.argmax(pred)]
    prob = float(np.max(pred))
    return img, label, prob

def show_prediction(img, label, prob):
    """Display the image with prediction using matplotlib."""
    plt.imshow(img)
    plt.title(f"Prediction: {label} ({prob:.2f})")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1.5)  # Display each image for 1.5 seconds
    plt.clf()

if __name__ == "__main__":
    test_dir = "data/test"
    
    for cls in class_names:
        folder_path = os.path.join(test_dir, cls)
        print(f"\nPredicting images in '{folder_path}':")
        for fname in os.listdir(folder_path):
            file_path = os.path.join(folder_path, fname)
            if os.path.isfile(file_path):
                img, label, prob = predict_image(file_path)
                print(f"{fname} -> Prediction: {label}, Probability: {prob:.4f}")
                show_prediction(img, label, prob)
