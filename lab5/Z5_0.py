from keras.datasets.fashion_mnist import load_data
from keras.utils import to_categorical
from keras import Sequential, layers

from keras.saving import save_model

import tensorflow as tf
import os

import numpy as np
import matplotlib.pyplot as plt
import ssl

# ssl._create_default_https_context = ssl._create_unverified_context
(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2], 1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation="softmax")
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, epochs=5,
                     validation_data=(X_test, y_test), batch_size=32)

print("\n-----------------------\nEvaluate:")
result = model.evaluate(X_test,  y_test, verbose=2)

print("\n-----------------------\Test accuracy:")
plt.figure(figsize=(16,9))
plt.subplot(2,1,1)
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(2,1,2)
plt.title('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.suptitle(f"Test loss: {result[0]}\nTest accuracy: {result[1]}")
print('Test loss:', result[0])
print('Test accuracy:', result[1])

plt.tight_layout
plt.savefig('Z5_0.png')
plt.show()

if not os.path.exists('saved_model'):
    os.mkdir('saved_model')
model.save('saved_model/my_model.keras', overwrite=True)
save_model(model, 'saved_model/my_model.keras', overwrite=True, zipped=None)




