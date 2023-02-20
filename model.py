import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
import tensorflow as tf
import numpy as np
import seaborn as sns


# Load the dataset
data = pd.read_csv("merged_augmented_data.csv")


# Preprocess the data
features = data["Features"].str.replace("[","").str.replace("]","").str.split().apply(lambda x: [float(i) for i in x])
features = features.values.tolist()
features = StandardScaler().fit_transform(features)
labels = data["diagnosis"]
labels = LabelEncoder().fit_transform(labels)
labels = to_categorical(labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Design of the model
model = Sequential()
model.add(SimpleRNN(8, input_shape=(X_train.shape[1], 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# get predictions
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# get the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

# get classification report
cr = classification_report(np.argmax(y_test, axis=1), y_pred)

# get accuracy
acc = accuracy_score(np.argmax(y_test, axis=1), y_pred)

# print results
print("Confusion Matrix: \n", cm)
print("Classification Report: \n", cr)
print("Accuracy: ", acc)

ax = sns.heatmap(cm, annot=True, fmt='d')

print(model.summary())

# Save the model
model.save('Rnn8-dense128.h5')

