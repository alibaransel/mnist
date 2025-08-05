import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train_padded = tf.pad(x_train, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT')
x_test_padded = tf.pad(x_test, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT')

y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

print(f"Padded Training data shape: {x_train_padded.shape}")
print(f"Padded Testing data shape: {x_test_padded.shape}")
print(f"One-hot encoded training labels shape: {y_train_one_hot.shape}")
print("-" * 50)

model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1)),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(5, 5), activation='tanh'),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(120, activation='tanh'),
    Dense(84, activation='tanh'),
    Dense(10, activation='softmax')
])

model.summary()
print("-" * 50)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print("Training the LeNet-5 model...")

history = model.fit(
    x_train_padded, y_train_one_hot,
    epochs=50,
    batch_size=128,
    validation_data=(x_test_padded, y_test_one_hot)
)

print("-" * 50)
print("Evaluating the model on the test dataset...")

loss, accuracy = model.evaluate(x_test_padded, y_test_one_hot)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
