from keras.datasets.fashion_mnist import load_data
from keras.utils import to_categorical
from keras import Sequential, layers

from keras.saving import load_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import ssl
import numpy as np


# ssl._create_default_https_context = ssl._create_unverified_context
(X_train, y_train), (X_test, y_test) = load_data()

# i'm shrinking the dataset because it takes forever to train
X_train = X_train[:len(X_train) // 4]
y_train = y_train[:len(y_train) // 4]
X_test = X_test[:len(X_test) // 4]
y_test = y_test[:len(y_test) // 4]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2], 1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

loaded_model = load_model('saved_model/my_model.keras')

r = loaded_model.evaluate(X_test,  y_test, verbose=2)

extractor = Sequential(loaded_model.layers[:-2])  # without the dense layers
# extract features

X_train_extracted = extractor.predict(X_train)  
X_test_extracted = extractor.predict(X_test)

y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# train rf classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose=True)
rf_classifier.fit(X_train_extracted, y_train_labels)

# predict
y_pred = rf_classifier.predict(X_test_extracted)
accuracy = accuracy_score(y_test_labels, y_pred)

print("\n-----------------------\nLoaded model test loss and accuracy:")
print(f"{r[0]:.5f}, {r[1]:.5f}")
print("\n-----------------------\nRandom Forest accuracy:")
print(f"{accuracy:.5f}")


