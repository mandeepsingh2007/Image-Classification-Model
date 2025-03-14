ashion MNIST Classification using TensorFlow

📌 Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The model is trained to recognize clothing items such as shirts, trousers, and sneakers.

📂 Dataset

The dataset used is Fashion MNIST, which consists of:

60,000 training images

10,000 test images

Each image is 28x28 pixels in grayscale

🚀 Installation & Setup

Ensure you have Python and the necessary dependencies installed.

pip install tensorflow numpy matplotlib seaborn

🔥 Model Architecture

The CNN model consists of the following layers:

Conv2D (32 filters, 3x3 kernel, ReLU activation)

MaxPooling2D (2x2 pool size)

Conv2D (64 filters, 3x3 kernel, ReLU activation)

MaxPooling2D (2x2 pool size)

Flatten layer

Dense layer (128 neurons, ReLU activation)

Output layer (10 neurons for classification)

🏋️‍♂️ Training

The model is compiled and trained using:

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=5, validation_data=(test_X, test_Y))
model.fit(train_X, train_Y, epochs=10)

🎯 Evaluation

After training, the model achieves 89.5% accuracy on the test dataset.

test_loss, test_acc = model.evaluate(test_X, test_Y)
print('\nTest accuracy:', test_acc)

📊 Prediction & Visualization

The trained model makes predictions, and we visualize them using Matplotlib.

def show_prediction(index):
    plt.imshow(test_images[index].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted: {np.argmax(predictions[index])} | Actual: {test_Y[index]}")
    plt.show()

show_prediction(1)

🏆 Results

The model successfully classifies clothing items with high accuracy. Sample predictions can be visualized using the function above.

📜 License

This project is open-source and available for learning and modification.
