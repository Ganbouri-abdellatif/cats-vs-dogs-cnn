# 🐱🐶 Cats vs Dogs Classifier

This project implements a **Convolutional Neural Network (CNN)** in TensorFlow/Keras to classify images of cats and dogs.  
The dataset is [Microsoft Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765).
 
📊 Dataset

Around 300 images used in this demo (150 cats + 150 dogs).

If you want to train a highly accurate model, use the full Kaggle Cats vs Dogs dataset (~25,000 images).

The dataset is automatically split into:

train/ (80%)

valid/ (10%)

test/ (10%)

🏗️ Model Architecture

The CNN model is defined in model.py:

Convolutional layers with increasing filters (32 → 64 → 128 → 256)

MaxPooling layers for downsampling

Global Average Pooling before Dense layers

Dense layer (1024 neurons, ReLU activation)

Output layer (Softmax for 2 classes: Cat & Dog)

Optimizer: RMSProp (lr=0.001)
Loss: Sparse Categorical Crossentropy
Metrics: Accuracy

🚀 Training

Run the training script:

python src/train.py


During training:

- Data is split into train/valid/test automatically.

- Model trains for 10 epochs by default.

- A plot of accuracy & loss curves is generated.

The trained model is saved as:

cnn_cats_dogs.h5

🔮 Prediction

To test the model on a single image:

python src/predict.py

example output:

Prediction: Dog, Probability: 0.9842

It also displays the image with its predicted label using Matplotlib.

💡 Future Improvements

Use data augmentation to improve generalization.

Train with transfer learning (ResNet, MobileNet, VGG16).

Deploy the model using Flask/Streamlit for a web demo.

Train on GPU for faster performance.


🧑‍💻 Author

Ganbouri Abdellatif
📧 abdellatifganbouri@gmail.com

🔗 GitHub Profile

📜 License

This project is licensed under the MIT License.



